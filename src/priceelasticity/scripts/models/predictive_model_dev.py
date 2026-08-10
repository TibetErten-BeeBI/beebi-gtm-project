import sys
import gc
import math
import inspect
from typing import Dict, List, Tuple

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. Spark setup
# ============================================================

spark = SparkSession.builder.getOrCreate()


# ============================================================
# 2. Parameter helper
# ============================================================

def get_param(param_name: str, default_value: str = "") -> str:
    """
    Reads parameter from:
    1. Python file Job arguments:
       --param_name value
       --param_name=value
    2. Databricks widgets
    3. Default value
    """
    arg_key = f"--{param_name}"

    if arg_key in sys.argv:
        arg_index = sys.argv.index(arg_key)
        if arg_index + 1 < len(sys.argv):
            return sys.argv[arg_index + 1].strip()

    for arg in sys.argv:
        arg = str(arg).strip()
        if arg.startswith(arg_key + "="):
            return arg.split("=", 1)[1].strip()

    try:
        dbutils.widgets.text(param_name, default_value)
        return dbutils.widgets.get(param_name).strip()
    except Exception:
        return default_value


# ============================================================
# 3. MLflow experiment setup
# ============================================================

MLFLOW_EXPERIMENT_NAME = get_param(
    "mlflow_experiment_name",
    "/Users/vadali.tejasviram@beebi-consulting.com/PE_MDO_Model_Experiments"
)

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# ============================================================
# 4. Project setup and optional ARMA import
# ============================================================

try:
    user_email = (
        dbutils.notebook.entry_point
        .getDbutils()
        .notebook()
        .getContext()
        .userName()
        .get()
    )
    PROJECT_ROOT = f"/Workspace/Users/{user_email}/PE_work"
except Exception:
    PROJECT_ROOT = "/Workspace/Users/vadali.tejasviram@beebi-consulting.com/PE_work"

MODEL_DIR = f"{PROJECT_ROOT}/src/priceelasticity/scripts/models"

for path in [PROJECT_ROOT, MODEL_DIR]:
    if path not in sys.path:
        sys.path.append(path)

try:
    from arma_residual_model import ARMAResidualModel
    ARMA_AVAILABLE = True
    ARMA_IMPORT_ERROR = None
    print("SUCCESS: ARMAResidualModel imported in predictive model.")
except Exception as exc:
    ARMA_AVAILABLE = False
    ARMA_IMPORT_ERROR = f"{type(exc).__name__}: {str(exc)}"
    print("WARNING: ARMAResidualModel import failed.")
    print(ARMA_IMPORT_ERROR)


# ============================================================
# 5. Table config - generic table naming
# ============================================================

OUTPUT_SCHEMA = get_param("output_schema", "workspace.default")
TABLE_PREFIX = get_param("table_prefix", "")


def table_name(base_name: str) -> str:
    if TABLE_PREFIX:
        return f"{OUTPUT_SCHEMA}.{TABLE_PREFIX}_{base_name}"
    return f"{OUTPUT_SCHEMA}.{base_name}"


def view_name(base_name: str) -> str:
    if TABLE_PREFIX:
        return f"{TABLE_PREFIX}_{base_name}"
    return base_name


FEATURE_TABLE = table_name("pe_causal_features_dev")

SALES_MIXEDLINEAR_COUNTERFACTUAL_TABLE = table_name("pe_sales_mixedlinear_counterfactual")
PROBABILITY_MIXEDLINEAR_COUNTERFACTUAL_TABLE = table_name("pe_probability_mixedlinear_counterfactual")

# This table is created by causal_model_dev.py.
# It contains product-store level discount coefficients.
SALES_DISCOUNT_COEFFICIENT_TABLE = table_name("pe_sales_discount_coefficients")

PREDICTIVE_TRAINING_TABLE = table_name("pe_predictive_training_features")

SALES_PREDICTION_OUTPUT_TABLE = table_name("pe_sales_prediction_output")
PROBABILITY_PREDICTION_OUTPUT_TABLE = table_name("pe_probability_prediction_output")
PREDICTIVE_SCENARIO_OUTPUT_TABLE = table_name("pe_predictive_scenario_output")
PREDICTIVE_SUMMARY_TABLE = table_name("pe_predictive_model_summary")

SALES_PREDICTION_OUTPUT_VIEW = view_name("pe_sales_prediction_output_view")
PROBABILITY_PREDICTION_OUTPUT_VIEW = view_name("pe_probability_prediction_output_view")
PREDICTIVE_SCENARIO_OUTPUT_VIEW = view_name("pe_predictive_scenario_output_view")

RANDOM_SEED = 42

PROBABILITY_FLOOR = 0.01
PROBABILITY_CEILING = 0.99

# Business safety cap:
# Maximum allowed sales scenario uplift on log scale.
# exp(1.20) is about 3.32x max uplift from discount.
MAX_SALES_SCENARIO_LOG_UPLIFT = 1.20

# MDO / scenario generation config:
# This controls the maximum markdown scenario discount generated.
# Example: 0.30 means do not generate scenarios above 30%.
MAX_SCENARIO_DISCOUNT = float(get_param("max_discount", "0.20"))


# ============================================================
# Daily ARMA residual model settings
# ============================================================

ARMA_MIN_HISTORY_POINTS = int(get_param("arma_min_history_points", "28"))

MAX_SALES_ARMA_CORRECTION = float(get_param("max_sales_arma_correction", "0.75"))
ARMA_PHI_FLOOR = float(get_param("arma_phi_floor", "-0.80"))
ARMA_PHI_CEILING = float(get_param("arma_phi_ceiling", "0.80"))

MIN_RMSE_IMPROVEMENT_TO_USE_ARMA = float(
    get_param("min_rmse_improvement_to_use_arma", "0.0")
)


# ============================================================
# 6. Memory cleanup
# ============================================================

def cleanup_memory():
    try:
        gc.collect()
    except Exception as exc:
        print("Python garbage collection skipped:", exc)


# ============================================================
# 7. Config
# ============================================================

def get_common_config() -> Dict:
    return {
        "date_col": "date",
        "week_col": "wm_yr_wk",
        "article_col": "pe_article",
        "store_col": "pe_store_group",
        "group_col": "pe_article_store_group",

        # Discount scenarios are parameterized so they can be passed from
        # Databricks job parameters now, and from UI in future.
        "scenario_discounts": [
            float(x.strip())
            for x in get_param(
                "allowed_scenario_discounts",
                "0,0.05,0.075,0.10,0.15,0.20"
            ).split(",")
            if x.strip() != ""
        ],

        "identity_cols": [
            "base_data_grain",

            "date",
            "week_start_date",
            "wm_yr_wk",
            "calendar_month",
            "calendar_year",
            "calendar_day_id",
            "weekday",
            "wday",
            "day_of_week",
            "day_of_month",
            "week_of_year",
            "is_weekend",

            "pe_article",
            "pe_store_group",
            "pe_article_store_group",

            "pe_quantity",
            "pe_unit_price",
            "pe_actual_retail_price",
            "pe_average_zone_retail_price",
            "discount",

            "valid_price_flag",
            "valid_for_price_elasticity",
            "valid_for_mdo_input",

            # Rule 30 - Minimum History Rule:
            # Carry history_days into predictive output so MDO can reject rows with insufficient history.
            "history_days",

            # Rule 31 - Price Variation Rule:
            # Carry price_variation into predictive output so MDO can reject low price movement rows.
            "price_variation",

            # Rule 32 - Discount Variation Rule:
            # Carry discount_variation into predictive output so MDO can reject low discount movement rows.
            "discount_variation",

            # Rule 33 - Low Confidence Prediction Rule:
            # Carry prediction_confidence into predictive output when available.
            "prediction_confidence",

            "pe_store_stock_quantity",
            "pe_inventory_onhand_quantity",
            "pe_store_stock_quantity_for_model",
            "pe_inventory_onhand_quantity_for_model",
            "stock_available_flag",
            "inventory_available_flag",

            "pe_category",
            "pe_product_type",
            "pe_product_division",
            "pe_country",
        ],

        "train_test_split_ratio": 0.80,
    }


def get_sales_predictive_config() -> Dict:
    config = get_common_config()

    config.update({
        "branch_name": "sales",
        "model_name": "sales_daily_deuplifted_predictive_model",

        "label_col": "sales_deuplifted_log_quantity",
        "raw_target_col": "sold_qty_log",

        "training_filter_col": "valid_for_sales_training",
        "scoring_filter_col": "valid_for_price_elasticity",

        "prediction_col": "predicted_deuplifted_log_quantity",
        "output_table": SALES_PREDICTION_OUTPUT_TABLE,
        "output_view": SALES_PREDICTION_OUTPUT_VIEW,
    })

    return config


def get_probability_predictive_config() -> Dict:
    config = get_common_config()

    config.update({
        "branch_name": "probability",
        "model_name": "probability_daily_deuplifted_predictive_model",

        "label_col": "probability_deuplifted_target",
        "raw_target_col": "probability_target",

        "training_filter_col": "valid_for_probability_training_with_price",
        "scoring_filter_col": "valid_for_probability_training_with_price",

        "prediction_col": "predicted_deuplifted_probability_raw",
        "output_table": PROBABILITY_PREDICTION_OUTPUT_TABLE,
        "output_view": PROBABILITY_PREDICTION_OUTPUT_VIEW,
    })

    return config


# ============================================================
# 8. Helpers
# ============================================================

def table_exists(table_name: str) -> bool:
    try:
        spark.table(table_name).limit(1).count()
        return True
    except Exception:
        return False


def require_table(table_name: str) -> None:
    if not table_exists(table_name):
        raise ValueError(f"Required table not found: {table_name}")

    print(f"Using table: {table_name}")


def require_columns(df, required_cols: List[str], label: str) -> None:
    missing_cols = [
        col_name
        for col_name in required_cols
        if col_name not in df.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing required columns in {label}: {missing_cols}")


def get_existing_cols(df, candidate_cols: List[str]) -> List[str]:
    return [
        col_name
        for col_name in candidate_cols
        if col_name in df.columns
    ]


def ensure_column(df, col_name: str, default_value=None, cast_type=None):
    if col_name in df.columns:
        return df

    expr = F.lit(default_value)

    if cast_type:
        expr = expr.cast(cast_type)

    return df.withColumn(col_name, expr)


def polynomial_expr(discount_col: str, coefs: List[float]):
    """
    Global polynomial expression.

    Kept for probability branch and fallback compatibility.
    Sales branch uses row_polynomial_expr().
    """

    expr = F.lit(0.0)

    for idx, coef in enumerate(coefs, start=1):
        expr = expr + F.lit(float(coef)) * F.pow(
            F.col(discount_col).cast("double"),
            F.lit(idx)
        )

    return expr


COEFFICIENT_COLS = [
    "discount_effect_coefficient_1",
    "discount_effect_coefficient_2",
    "discount_effect_coefficient_3",
    "discount_effect_coefficient_4",
]


def row_polynomial_expr(discount_col: str):
    """
    Row-level discount effect.

    Uses coefficient columns joined by pe_article_store_group.
    """

    d = F.col(discount_col).cast("double")

    return (
        F.coalesce(F.col("discount_effect_coefficient_1"), F.lit(0.0)) * d
        + F.coalesce(F.col("discount_effect_coefficient_2"), F.lit(0.0)) * F.pow(d, F.lit(2))
        + F.coalesce(F.col("discount_effect_coefficient_3"), F.lit(0.0)) * F.pow(d, F.lit(3))
        + F.coalesce(F.col("discount_effect_coefficient_4"), F.lit(0.0)) * F.pow(d, F.lit(4))
    )


def row_log_price_elasticity_expr(price_ratio_col: str):
    """
    Correct elasticity-based sales uplift.

    Formula:
        effect = elasticity * log(price_ratio)

    If:
        elasticity < 0
        price_ratio < 1

    Then:
        effect is positive, which increases scenario quantity.
    """

    return (
        F.coalesce(F.col("estimated_elasticity"), F.lit(-1.0))
        * F.log(F.col(price_ratio_col).cast("double"))
    )


def load_sales_discount_coefficient_df():
    """
    Loads product-store level sales discount coefficients.

    Important:
    Rename history/variation columns to avoid duplicate column names
    after joining with feature_df.
    """

    require_table(SALES_DISCOUNT_COEFFICIENT_TABLE)

    coef_df = spark.table(SALES_DISCOUNT_COEFFICIENT_TABLE)

    require_columns(
        coef_df,
        ["pe_article_store_group"] + COEFFICIENT_COLS,
        "sales discount coefficient table"
    )

    select_exprs = [
        F.col("pe_article_store_group"),
        F.col("discount_effect_coefficient_1"),
        F.col("discount_effect_coefficient_2"),
        F.col("discount_effect_coefficient_3"),
        F.col("discount_effect_coefficient_4"),
    ]

    if "coefficient_level" in coef_df.columns:
        select_exprs.append(F.col("coefficient_level"))

    if "coefficient_source" in coef_df.columns:
        select_exprs.append(F.col("coefficient_source"))

    if "model_status" in coef_df.columns:
        select_exprs.append(F.col("model_status"))

    if "random_effect_applied" in coef_df.columns:
        select_exprs.append(F.col("random_effect_applied"))

    if "history_rows" in coef_df.columns:
        select_exprs.append(
            F.col("history_rows").alias("coefficient_history_rows")
        )

    if "discount_variation_count" in coef_df.columns:
        select_exprs.append(
            F.col("discount_variation_count").alias("coefficient_discount_variation_count")
        )

    if "global_discount_effect_coefficient_1" in coef_df.columns:
        select_exprs.append(F.col("global_discount_effect_coefficient_1"))

    if "random_discount_effect_coefficient_1" in coef_df.columns:
        select_exprs.append(F.col("random_discount_effect_coefficient_1"))

    if "estimated_elasticity" in coef_df.columns:
        select_exprs.append(F.col("estimated_elasticity"))

    if "elasticity_source" in coef_df.columns:
        select_exprs.append(F.col("elasticity_source"))

    if "elasticity_status" in coef_df.columns:
        select_exprs.append(F.col("elasticity_status"))

    result_df = (
        coef_df
        .select(*select_exprs)
        .dropDuplicates(["pe_article_store_group"])
    )

    print("Loaded sales discount coefficient table:", SALES_DISCOUNT_COEFFICIENT_TABLE)

    return result_df


def sigmoid_expr(col_expr):
    return F.lit(1.0) / (F.lit(1.0) + F.exp(-col_expr))


def clip_probability_expr(col_expr):
    return (
        F.when(col_expr < F.lit(PROBABILITY_FLOOR), F.lit(PROBABILITY_FLOOR))
         .when(col_expr > F.lit(PROBABILITY_CEILING), F.lit(PROBABILITY_CEILING))
         .otherwise(col_expr)
    )


def logit_expr(prob_col_expr):
    return F.log(prob_col_expr / (F.lit(1.0) - prob_col_expr))


def validate_daily_feature_table(feature_df):
    """
    Daily safety check.

    Expected feature grain:
        date + pe_article + pe_store_group
    """

    if "base_data_grain" in feature_df.columns:
        grains = [
            row["base_data_grain"]
            for row in feature_df.select("base_data_grain").distinct().collect()
        ]

        print("Feature table grain values:", grains)

        if "product_store_day" not in grains:
            raise ValueError(
                "Predictive model expects daily features: base_data_grain = product_store_day. "
                f"Found: {grains}. Run daily base data and daily feature engineering first."
            )

    duplicate_count = (
        feature_df
        .groupBy("date", "pe_article", "pe_store_group")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Feature table has duplicate product-store-date keys: {duplicate_count}. "
            "Expected one row per date + product + store."
        )


def load_discount_coefficients(counterfactual_table: str, label: str) -> List[float]:
    """
    Loads one global coefficient row.

    This is still used for probability branch.
    """

    require_table(counterfactual_table)

    df = spark.table(counterfactual_table)

    required_cols = [
        "discount_effect_coefficient_1",
        "discount_effect_coefficient_2",
        "discount_effect_coefficient_3",
        "discount_effect_coefficient_4",
    ]

    require_columns(df, required_cols, label)

    row = (
        df
        .select(*required_cols)
        .where(F.col("discount_effect_coefficient_1").isNotNull())
        .limit(1)
        .collect()
    )

    if not row:
        raise ValueError(f"No coefficient row found in {counterfactual_table}")

    coefs = [
        float(row[0]["discount_effect_coefficient_1"]),
        float(row[0]["discount_effect_coefficient_2"]),
        float(row[0]["discount_effect_coefficient_3"]),
        float(row[0]["discount_effect_coefficient_4"]),
    ]

    print(label, "discount coefficients:", coefs)

    return coefs


def create_scenario_df(scenario_discounts: List[float]):
    rows = [(float(x),) for x in scenario_discounts]
    return spark.createDataFrame(rows, ["scenario_discount"])


def build_time_split(df, config: Dict):
    """
    Daily time split.
    """

    date_col = config["date_col"]
    split_ratio = config["train_test_split_ratio"]

    dates = [
        row[date_col]
        for row in (
            df
            .select(date_col)
            .where(F.col(date_col).isNotNull())
            .distinct()
            .orderBy(date_col)
            .collect()
        )
    ]

    if len(dates) < 3:
        print("Not enough dates for daily time split. Using random split.")
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
        return train_df, test_df, None

    cutoff_index = max(1, int(len(dates) * split_ratio) - 1)
    cutoff_date = dates[cutoff_index]

    train_df = df.filter(F.col(date_col) <= F.lit(cutoff_date))
    test_df = df.filter(F.col(date_col) > F.lit(cutoff_date))

    if test_df.count() == 0:
        print("Daily time split produced empty test set. Using random split.")
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
        return train_df, test_df, None

    print("Time split cutoff date:", cutoff_date)

    return train_df, test_df, str(cutoff_date)


def add_hierarchical_mean_prediction(train_df, score_df, label_col: str, prediction_col: str, config: Dict):
    """
    Spark-only baseline predictive model.

    Prediction hierarchy:
        product-store average
        article average
        store average
        global average
    """

    group_col = config["group_col"]
    article_col = config["article_col"]
    store_col = config["store_col"]

    global_row = (
        train_df
        .agg(F.avg(F.col(label_col)).alias("global_prediction"))
        .collect()
    )

    if not global_row or global_row[0]["global_prediction"] is None:
        raise ValueError(f"Cannot create prediction: global average is null for {label_col}")

    global_prediction = float(global_row[0]["global_prediction"])

    group_avg_df = (
        train_df
        .groupBy(group_col)
        .agg(F.avg(F.col(label_col)).alias("_group_prediction"))
    )

    article_avg_df = (
        train_df
        .groupBy(article_col)
        .agg(F.avg(F.col(label_col)).alias("_article_prediction"))
    )

    store_avg_df = (
        train_df
        .groupBy(store_col)
        .agg(F.avg(F.col(label_col)).alias("_store_prediction"))
    )

    scored_df = (
        score_df
        .join(group_avg_df, on=group_col, how="left")
        .join(article_avg_df, on=article_col, how="left")
        .join(store_avg_df, on=store_col, how="left")
        .withColumn(
            prediction_col,
            F.coalesce(
                F.col("_group_prediction"),
                F.col("_article_prediction"),
                F.col("_store_prediction"),
                F.lit(global_prediction),
            )
        )
        .drop("_group_prediction", "_article_prediction", "_store_prediction")
    )

    return scored_df, global_prediction


def build_sales_residual_training_df(
    train_df,
    label_col: str,
    prediction_col: str,
    config: Dict,
):
    """
    Build residual training data for daily ARMA.
    """

    group_col = config["group_col"]
    date_col = config["date_col"]
    week_col = config["week_col"]

    train_predictions, _ = add_hierarchical_mean_prediction(
        train_df=train_df,
        score_df=train_df,
        label_col=label_col,
        prediction_col=prediction_col,
        config=config,
    )

    select_exprs = [
        F.col(group_col),
        F.col(date_col),
        F.col("sales_residual"),
    ]

    if week_col in train_predictions.columns:
        select_exprs.insert(2, F.col(week_col))

    residual_df = (
        train_predictions
        .withColumn(
            "sales_residual",
            F.col(label_col) - F.col(prediction_col)
        )
        .select(*select_exprs)
        .filter(F.col("sales_residual").isNotNull())
    )

    return residual_df


def create_arma_model(config: Dict):
    """
    Create ARMAResidualModel with daily date support.
    """

    if not ARMA_AVAILABLE:
        return None

    init_signature = inspect.signature(ARMAResidualModel.__init__)
    params = init_signature.parameters

    kwargs = {
        "group_col": config["group_col"],
        "residual_col": "sales_residual",
        "min_history_points": ARMA_MIN_HISTORY_POINTS,
        "max_abs_correction": MAX_SALES_ARMA_CORRECTION,
        "phi_floor": ARMA_PHI_FLOOR,
        "phi_ceiling": ARMA_PHI_CEILING,
    }

    if "date_col" in params:
        kwargs["date_col"] = config["date_col"]
    elif "time_col" in params:
        kwargs["time_col"] = config["date_col"]
    elif "week_col" in params:
        kwargs["week_col"] = config["date_col"]

    return ARMAResidualModel(**kwargs)


def apply_sales_arma_correction(
    scored_df,
    arma_model,
    prediction_col: str,
):
    """
    Apply ARMA residual correction to baseline sales prediction.
    """

    scored_with_arma_df = arma_model.transform(
        score_df=scored_df,
        output_col="arma_residual_correction",
    )

    required_defaults = [
        ("arma_history_points", 0, "int"),
        ("arma_mean_residual", 0.0, "double"),
        ("arma_last_residual", 0.0, "double"),
        ("arma_phi_raw", 0.0, "double"),
        ("arma_phi", 0.0, "double"),
        ("arma_residual_correction_raw", 0.0, "double"),
        ("arma_residual_correction", 0.0, "double"),
        ("arma_min_history_points", ARMA_MIN_HISTORY_POINTS, "int"),
        ("arma_max_abs_correction", MAX_SALES_ARMA_CORRECTION, "double"),
        ("arma_last_date", None, "date"),
        ("arma_last_week", None, "string"),
    ]

    for col_name, default_value, cast_type in required_defaults:
        scored_with_arma_df = ensure_column(
            scored_with_arma_df,
            col_name,
            default_value,
            cast_type
        )

    scored_with_arma_df = (
        scored_with_arma_df
        .withColumn(
            "predicted_deuplifted_log_quantity_corrected",
            F.col(prediction_col) + F.col("arma_residual_correction")
        )
        .withColumn(
            "predicted_deuplifted_quantity_corrected",
            F.exp(F.col("predicted_deuplifted_log_quantity_corrected"))
        )
        .withColumn(
            "sales_residual_correction_raw",
            F.col("arma_residual_correction_raw")
        )
        .withColumn(
            "sales_residual_correction",
            F.col("arma_residual_correction")
        )
        .withColumn(
            "sales_residual_points",
            F.col("arma_history_points")
        )
        .withColumn(
            "sales_residual_lookback_days",
            F.col("arma_min_history_points")
        )
        .withColumn(
            "sales_residual_lookback_start_date",
            F.col("arma_last_date")
        )
        .withColumn(
            "sales_residual_lookback_weeks",
            F.col("arma_min_history_points")
        )
        .withColumn(
            "sales_residual_lookback_start_week",
            F.col("arma_last_week")
        )
    )

    return scored_with_arma_df


def disable_sales_arma_correction(
    scored_df,
    prediction_col: str,
):
    """
    Keep ARMA/residual columns for schema compatibility,
    but force corrected prediction to equal baseline prediction.
    """

    disabled_df = (
        scored_df
        .withColumn("arma_history_points", F.lit(0))
        .withColumn("arma_mean_residual", F.lit(0.0))
        .withColumn("arma_last_residual", F.lit(0.0))
        .withColumn("arma_phi_raw", F.lit(0.0))
        .withColumn("arma_phi", F.lit(0.0))
        .withColumn("arma_residual_correction_raw", F.lit(0.0))
        .withColumn("arma_residual_correction", F.lit(0.0))
        .withColumn("arma_min_history_points", F.lit(ARMA_MIN_HISTORY_POINTS))
        .withColumn("arma_max_abs_correction", F.lit(MAX_SALES_ARMA_CORRECTION))
        .withColumn("arma_last_date", F.lit(None).cast("date"))
        .withColumn("arma_last_week", F.lit(None).cast("string"))

        .withColumn("sales_residual_correction_raw", F.lit(0.0))
        .withColumn("sales_residual_correction", F.lit(0.0))
        .withColumn("sales_residual_points", F.lit(0))
        .withColumn("sales_residual_lookback_days", F.lit(ARMA_MIN_HISTORY_POINTS))
        .withColumn("sales_residual_lookback_start_date", F.lit(None).cast("date"))

        .withColumn("sales_residual_lookback_weeks", F.lit(ARMA_MIN_HISTORY_POINTS))
        .withColumn("sales_residual_lookback_start_week", F.lit(None).cast("string"))

        .withColumn(
            "predicted_deuplifted_log_quantity_corrected",
            F.col(prediction_col)
        )
        .withColumn(
            "predicted_deuplifted_quantity_corrected",
            F.exp(F.col(prediction_col))
        )
    )

    return disabled_df


def compute_regression_metrics(scored_df, label_col: str, prediction_col: str) -> Dict:
    clean_df = (
        scored_df
        .filter(F.col(label_col).isNotNull())
        .filter(F.col(prediction_col).isNotNull())
    )

    base_stats = (
        clean_df
        .agg(
            F.count("*").alias("rows"),
            F.avg(F.col(label_col)).alias("label_mean"),
            F.avg(F.pow(F.col(label_col) - F.col(prediction_col), 2)).alias("mse"),
            F.avg(F.abs(F.col(label_col) - F.col(prediction_col))).alias("mae"),
        )
        .collect()[0]
    )

    rows = int(base_stats["rows"])
    label_mean = float(base_stats["label_mean"]) if base_stats["label_mean"] is not None else 0.0
    mse = float(base_stats["mse"]) if base_stats["mse"] is not None else None
    mae = float(base_stats["mae"]) if base_stats["mae"] is not None else None
    rmse = math.sqrt(mse) if mse is not None else None

    ss_stats = (
        clean_df
        .agg(
            F.sum(F.pow(F.col(label_col) - F.col(prediction_col), 2)).alias("sse"),
            F.sum(F.pow(F.col(label_col) - F.lit(label_mean), 2)).alias("sst"),
        )
        .collect()[0]
    )

    sse = float(ss_stats["sse"]) if ss_stats["sse"] is not None else None
    sst = float(ss_stats["sst"]) if ss_stats["sst"] is not None else None

    if sse is not None and sst is not None and sst > 0:
        r2 = 1.0 - (sse / sst)
    else:
        r2 = None

    return {
        "rows": rows,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def metric_rows(branch_name: str, model_name: str, metrics: Dict) -> List[Tuple[str, str, str, str]]:
    rows = []

    for metric_name, metric_value in metrics.items():
        rows.append(
            (
                branch_name,
                model_name,
                metric_name,
                str(metric_value),
            )
        )

    return rows


# ============================================================
# 9. Build de-uplifted predictive training features
# ============================================================

def build_predictive_training_features():
    print("==================================================")
    print("Building DAILY de-uplifted predictive training features")
    print("==================================================")

    require_table(FEATURE_TABLE)
    require_table(SALES_MIXEDLINEAR_COUNTERFACTUAL_TABLE)
    require_table(PROBABILITY_MIXEDLINEAR_COUNTERFACTUAL_TABLE)
    require_table(SALES_DISCOUNT_COEFFICIENT_TABLE)

    feature_df = spark.table(FEATURE_TABLE)

    require_columns(
        feature_df,
        [
            "date",
            "wm_yr_wk",
            "pe_article",
            "pe_store_group",
            "pe_article_store_group",
            "discount",
            "sold_qty_log",
            "probability_target",
            "probability_target_smoothed",
            "probability_logit_target",
            "valid_for_sales_training",
            "valid_for_probability_training_with_price",
            "valid_for_price_elasticity",
        ],
        "feature table"
    )

    validate_daily_feature_table(feature_df)

    sales_coef_df = load_sales_discount_coefficient_df()

    sales_coefs = [0.0, 0.0, 0.0, 0.0]

    probability_coefs = load_discount_coefficients(
        PROBABILITY_MIXEDLINEAR_COUNTERFACTUAL_TABLE,
        "probability mixed-linear"
    )

    if probability_coefs[0] < 0:
        print("WARNING: Negative probability discount coefficient found.")
        print("Original probability coefficients:", probability_coefs)
        print("Setting probability discount coefficients to zero for predictive/MDO safety.")
        probability_coefs = [0.0, 0.0, 0.0, 0.0]

    predictive_df = (
        feature_df
        .join(
            sales_coef_df,
            on="pe_article_store_group",
            how="left"
        )
        .withColumn(
            "current_price_ratio",
            F.when(
                (F.col("pe_average_zone_retail_price").isNotNull()) &
                (F.col("pe_average_zone_retail_price") > 0) &
                (F.col("pe_unit_price").isNotNull()) &
                (F.col("pe_unit_price") > 0),
                F.col("pe_unit_price") / F.col("pe_average_zone_retail_price")
            ).otherwise(
                F.lit(1.0) - F.col("discount")
            )
        )
        .withColumn(
            "current_price_ratio",
            F.when(F.col("current_price_ratio") <= 0, F.lit(1.0))
             .otherwise(F.col("current_price_ratio"))
        )
        .withColumn(
            "sales_current_discount_effect",
            row_log_price_elasticity_expr("current_price_ratio")
        )
        .withColumn(
            "probability_current_discount_effect",
            polynomial_expr("discount", probability_coefs)
        )
        .withColumn(
            "sales_deuplifted_log_quantity",
            F.when(
                F.col("sold_qty_log").isNotNull(),
                F.col("sold_qty_log") - F.col("sales_current_discount_effect")
            ).otherwise(F.lit(None).cast("double"))
        )
        .withColumn(
            "probability_deuplifted_logit",
            F.when(
                F.col("probability_logit_target").isNotNull(),
                F.col("probability_logit_target") - F.col("probability_current_discount_effect")
            ).otherwise(F.lit(None).cast("double"))
        )
        .withColumn(
            "probability_deuplifted_target",
            sigmoid_expr(F.col("probability_deuplifted_logit"))
        )
        .withColumn("sales_uplift_source_table", F.lit(SALES_DISCOUNT_COEFFICIENT_TABLE))
        .withColumn("probability_uplift_source_table", F.lit(PROBABILITY_MIXEDLINEAR_COUNTERFACTUAL_TABLE))
        .withColumn("predictive_training_created_at", F.current_timestamp())
    )

    (
        predictive_df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(PREDICTIVE_TRAINING_TABLE)
    )

    print("Saved predictive training table:", PREDICTIVE_TRAINING_TABLE)

    print("Validation: predictive training feature counts")
    display(
        spark.sql(f"""
            SELECT
                base_data_grain,
                COUNT(*) AS rows,
                COUNT(DISTINCT pe_article) AS products,
                COUNT(DISTINCT pe_store_group) AS stores,
                COUNT(DISTINCT pe_article_store_group) AS product_store_groups,
                COUNT(DISTINCT date) AS days,
                COUNT(DISTINCT wm_yr_wk) AS weeks,
                SUM(valid_for_sales_training) AS sales_training_rows,
                SUM(valid_for_probability_training_with_price) AS probability_training_rows
            FROM {PREDICTIVE_TRAINING_TABLE}
            GROUP BY base_data_grain
        """)
    )

    print("Validation: sales coefficient usage")
    display(
        spark.sql(f"""
            SELECT
                coefficient_level,
                model_status,
                COUNT(*) AS rows,
                AVG(sales_current_discount_effect) AS avg_sales_current_discount_effect,
                AVG(discount_effect_coefficient_1) AS avg_coef_1,
                MIN(discount_effect_coefficient_1) AS min_coef_1,
                MAX(discount_effect_coefficient_1) AS max_coef_1,
                STDDEV(discount_effect_coefficient_1) AS stddev_coef_1
            FROM {PREDICTIVE_TRAINING_TABLE}
            GROUP BY coefficient_level, model_status
        """)
    )

    print("Validation: one row per product-store-date")
    display(
        spark.sql(f"""
            SELECT
                COUNT(*) AS duplicate_keys
            FROM (
                SELECT
                    date,
                    pe_article,
                    pe_store_group,
                    COUNT(*) AS rows_per_key
                FROM {PREDICTIVE_TRAINING_TABLE}
                GROUP BY date, pe_article, pe_store_group
                HAVING COUNT(*) > 1
            )
        """)
    )

    print("Validation: de-uplift probability distribution")
    display(
        spark.sql(f"""
            SELECT
                MIN(probability_deuplifted_logit) AS min_probability_deuplifted_logit,
                MAX(probability_deuplifted_logit) AS max_probability_deuplifted_logit,
                AVG(probability_deuplifted_logit) AS avg_probability_deuplifted_logit,
                MIN(probability_deuplifted_target) AS min_probability_deuplifted_target,
                MAX(probability_deuplifted_target) AS max_probability_deuplifted_target,
                AVG(probability_deuplifted_target) AS avg_probability_deuplifted_target
            FROM {PREDICTIVE_TRAINING_TABLE}
            WHERE valid_for_probability_training_with_price = 1
        """)
    )

    cleanup_memory()

    return PREDICTIVE_TRAINING_TABLE, sales_coefs, probability_coefs


# ============================================================
# 10. Train sales predictive model
# ============================================================

def train_sales_predictive_model(training_table_name: str):
    config = get_sales_predictive_config()

    print("==================================================")
    print("Training DAILY sales predictive model with optional ARMA residual correction")
    print("==================================================")

    label_col = config["label_col"]

    training_feature_df = spark.table(training_table_name)

    training_base_df = (
        training_feature_df
        .filter(F.col(config["training_filter_col"]) == 1)
        .filter(F.col(label_col).isNotNull())
    )

    scoring_base_df = (
        training_feature_df
        .filter(F.col(config["scoring_filter_col"]) == 1)
    )

    print("Sales training rows:", training_base_df.count())
    print("Sales scoring rows:", scoring_base_df.count())

    train_df, test_df, cutoff_date = build_time_split(training_base_df, config)

    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=config["model_name"]):
        train_count = train_df.count()
        test_count = test_df.count()
        scoring_count = scoring_base_df.count()

        mlflow.log_param("branch_name", config["branch_name"])
        mlflow.log_param("model_name", config["model_name"])
        mlflow.log_param("model_type", "daily_hierarchical_mean_deuplifted_with_optional_arma_residual")
        mlflow.log_param("label_col", label_col)
        mlflow.log_param("training_rows", train_count)
        mlflow.log_param("test_rows", test_count)
        mlflow.log_param("scoring_rows", scoring_count)
        mlflow.log_param("cutoff_date", cutoff_date)
        mlflow.log_param("daily_grain", "product_store_day")

        mlflow.log_param("arma_available", ARMA_AVAILABLE)
        mlflow.log_param("arma_import_error", ARMA_IMPORT_ERROR)
        mlflow.log_param("arma_min_history_points_days", ARMA_MIN_HISTORY_POINTS)
        mlflow.log_param("max_sales_arma_correction", MAX_SALES_ARMA_CORRECTION)
        mlflow.log_param("arma_phi_floor", ARMA_PHI_FLOOR)
        mlflow.log_param("arma_phi_ceiling", ARMA_PHI_CEILING)
        mlflow.log_param("min_rmse_improvement_to_use_arma", MIN_RMSE_IMPROVEMENT_TO_USE_ARMA)

        test_predictions, global_prediction = add_hierarchical_mean_prediction(
            train_df=train_df,
            score_df=test_df,
            label_col=label_col,
            prediction_col=config["prediction_col"],
            config=config,
        )

        metrics_eval = compute_regression_metrics(
            scored_df=test_predictions,
            label_col=label_col,
            prediction_col=config["prediction_col"],
        )

        rmse = metrics_eval["rmse"]
        mae = metrics_eval["mae"]
        r2 = metrics_eval["r2"]

        arma_model = None
        arma_failure_reason = None

        rmse_arma = None
        mae_arma = None
        r2_arma = None
        arma_rmse_improvement = None

        if ARMA_AVAILABLE:
            try:
                residual_train_df = build_sales_residual_training_df(
                    train_df=train_df,
                    label_col=label_col,
                    prediction_col=config["prediction_col"],
                    config=config,
                )

                arma_model = create_arma_model(config)

                if arma_model is None:
                    raise ValueError("ARMA model object could not be created.")

                arma_model.fit(residual_train_df)

                test_predictions_arma = apply_sales_arma_correction(
                    scored_df=test_predictions,
                    arma_model=arma_model,
                    prediction_col=config["prediction_col"],
                )

                metrics_eval_arma = compute_regression_metrics(
                    scored_df=test_predictions_arma,
                    label_col=label_col,
                    prediction_col="predicted_deuplifted_log_quantity_corrected",
                )

                rmse_arma = metrics_eval_arma["rmse"]
                mae_arma = metrics_eval_arma["mae"]
                r2_arma = metrics_eval_arma["r2"]

                if rmse is not None and rmse_arma is not None:
                    arma_rmse_improvement = float(rmse) - float(rmse_arma)

            except Exception as exc:
                arma_failure_reason = f"{type(exc).__name__}: {str(exc)}"
                print("WARNING: ARMA residual model failed. Falling back to baseline.")
                print(arma_failure_reason)
        else:
            arma_failure_reason = ARMA_IMPORT_ERROR

        use_sales_arma_correction = (
            arma_rmse_improvement is not None
            and arma_rmse_improvement > MIN_RMSE_IMPROVEMENT_TO_USE_ARMA
            and arma_model is not None
        )

        if use_sales_arma_correction:
            print("ARMA residual correction improved RMSE. Using ARMA-corrected sales baseline.")
        else:
            print("ARMA residual correction did not improve RMSE or failed. Keeping baseline sales prediction.")

        mlflow.log_param("global_prediction", global_prediction)
        mlflow.log_param("use_sales_arma_correction", use_sales_arma_correction)
        mlflow.log_param("arma_failure_reason", arma_failure_reason)

        if rmse is not None:
            mlflow.log_metric("rmse", rmse)
        if mae is not None:
            mlflow.log_metric("mae", mae)
        if r2 is not None:
            mlflow.log_metric("r2", r2)

        if rmse_arma is not None:
            mlflow.log_metric("rmse_arma_candidate", rmse_arma)
        if mae_arma is not None:
            mlflow.log_metric("mae_arma_candidate", mae_arma)
        if r2_arma is not None:
            mlflow.log_metric("r2_arma_candidate", r2_arma)
        if arma_rmse_improvement is not None:
            mlflow.log_metric("rmse_improvement_from_arma", arma_rmse_improvement)

        scored_df, _ = add_hierarchical_mean_prediction(
            train_df=train_df,
            score_df=scoring_base_df,
            label_col=label_col,
            prediction_col=config["prediction_col"],
            config=config,
        )

        if use_sales_arma_correction:
            scored_df = apply_sales_arma_correction(
                scored_df=scored_df,
                arma_model=arma_model,
                prediction_col=config["prediction_col"],
            )
        else:
            scored_df = disable_sales_arma_correction(
                scored_df=scored_df,
                prediction_col=config["prediction_col"],
            )

        scored_df = (
            scored_df
            .withColumn("use_sales_arma_correction", F.lit(1 if use_sales_arma_correction else 0))
            .withColumn("use_sales_residual_correction", F.lit(1 if use_sales_arma_correction else 0))
        )

    output_cols = get_existing_cols(scored_df, config["identity_cols"])

    coefficient_output_cols = get_existing_cols(
        scored_df,
        [
            "coefficient_level",
            "coefficient_source",
            "model_status",
            "random_effect_applied",
            "coefficient_history_rows",
            "coefficient_discount_variation_count",
            "discount_effect_coefficient_1",
            "discount_effect_coefficient_2",
            "discount_effect_coefficient_3",
            "discount_effect_coefficient_4",
            "global_discount_effect_coefficient_1",
            "random_discount_effect_coefficient_1",
            "estimated_elasticity",
            "elasticity_source",
            "elasticity_status",
            "current_price_ratio",
        ]
    )

    sales_output_df = (
        scored_df
        .withColumn(
            "predicted_deuplifted_quantity",
            F.exp(F.col(config["prediction_col"]))
        )
        .withColumn(
            "predicted_deuplifted_quantity",
            F.when(F.col("predicted_deuplifted_quantity") < 0, F.lit(0.0))
             .otherwise(F.col("predicted_deuplifted_quantity"))
        )
        .withColumn("prediction_branch", F.lit("sales"))
        .withColumn("prediction_model_name", F.lit(config["model_name"]))
        .withColumn("prediction_model_type", F.lit("daily_hierarchical_mean_deuplifted_with_optional_arma_residual"))
        .withColumn("model_created_at", F.current_timestamp())
        .select(
            *output_cols,
            *coefficient_output_cols,

            F.col(config["raw_target_col"]).alias("actual_log_quantity"),
            F.col("sales_current_discount_effect"),
            F.col(label_col).alias("sales_deuplifted_log_quantity"),
            F.col(config["prediction_col"]).alias("predicted_deuplifted_log_quantity"),
            F.col("predicted_deuplifted_quantity"),

            F.col("use_sales_arma_correction"),
            F.col("use_sales_residual_correction"),

            F.col("arma_history_points"),
            F.col("arma_mean_residual"),
            F.col("arma_last_residual"),
            F.col("arma_phi_raw"),
            F.col("arma_phi"),
            F.col("arma_residual_correction_raw"),
            F.col("arma_residual_correction"),
            F.col("arma_min_history_points"),
            F.col("arma_max_abs_correction"),
            F.col("arma_last_date"),
            F.col("arma_last_week"),

            F.col("sales_residual_correction_raw"),
            F.col("sales_residual_correction"),
            F.col("sales_residual_points"),
            F.col("sales_residual_lookback_days"),
            F.col("sales_residual_lookback_start_date"),

            F.col("sales_residual_lookback_weeks"),
            F.col("sales_residual_lookback_start_week"),

            F.col("predicted_deuplifted_log_quantity_corrected"),
            F.col("predicted_deuplifted_quantity_corrected"),

            "prediction_branch",
            "prediction_model_name",
            "prediction_model_type",
            "model_created_at"
        )
    )

    (
        sales_output_df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(config["output_table"])
    )

    sales_output_df.createOrReplaceTempView(config["output_view"])

    if use_sales_arma_correction:
        final_rmse = rmse_arma
        final_mae = mae_arma
        final_r2 = r2_arma
    else:
        final_rmse = rmse
        final_mae = mae
        final_r2 = r2

    metrics = {
        "model_type": "daily_hierarchical_mean_deuplifted_with_optional_arma_residual",
        "target_col": label_col,
        "training_filter_col": config["training_filter_col"],
        "scoring_filter_col": config["scoring_filter_col"],
        "train_rows": train_count,
        "test_rows": test_count,
        "scoring_rows": scoring_count,
        "cutoff_date": cutoff_date,
        "global_prediction": global_prediction,

        "arma_available": ARMA_AVAILABLE,
        "arma_import_error": ARMA_IMPORT_ERROR,
        "arma_failure_reason": arma_failure_reason,
        "arma_min_history_points_days": ARMA_MIN_HISTORY_POINTS,
        "max_sales_arma_correction": MAX_SALES_ARMA_CORRECTION,
        "arma_phi_floor": ARMA_PHI_FLOOR,
        "arma_phi_ceiling": ARMA_PHI_CEILING,
        "min_rmse_improvement_to_use_arma": MIN_RMSE_IMPROVEMENT_TO_USE_ARMA,
        "use_sales_arma_correction": use_sales_arma_correction,
        "use_sales_residual_correction": use_sales_arma_correction,

        "rmse_baseline": rmse,
        "mae_baseline": mae,
        "r2_baseline": r2,

        "rmse_arma_candidate": rmse_arma,
        "mae_arma_candidate": mae_arma,
        "r2_arma_candidate": r2_arma,
        "rmse_improvement_from_arma": arma_rmse_improvement,

        "rmse_corrected_candidate": rmse_arma,
        "mae_corrected_candidate": mae_arma,
        "r2_corrected_candidate": r2_arma,
        "rmse_improvement_from_residual": arma_rmse_improvement,

        "rmse": final_rmse,
        "mae": final_mae,
        "r2": final_r2,

        "sales_discount_coefficient_table": SALES_DISCOUNT_COEFFICIENT_TABLE,
        "output_table": config["output_table"],
    }

    print("Daily sales predictive model completed.")
    print("Sales output table:", config["output_table"])
    print("Baseline RMSE:", rmse)
    print("Baseline MAE:", mae)
    print("Baseline R2:", r2)
    print("ARMA candidate RMSE:", rmse_arma)
    print("ARMA candidate MAE:", mae_arma)
    print("ARMA candidate R2:", r2_arma)
    print("ARMA RMSE improvement:", arma_rmse_improvement)
    print("Use sales ARMA correction:", use_sales_arma_correction)
    print("Final RMSE:", final_rmse)
    print("Final MAE:", final_mae)
    print("Final R2:", final_r2)

    cleanup_memory()

    return metric_rows(
        config["branch_name"],
        config["model_name"],
        metrics
    )


# ============================================================
# 11. Train probability predictive model
# ============================================================

def train_probability_predictive_model(training_table_name: str):
    config = get_probability_predictive_config()

    print("==================================================")
    print("Training DAILY probability predictive model with Spark-only hierarchical means")
    print("==================================================")

    label_col = config["label_col"]

    training_feature_df = spark.table(training_table_name)

    training_base_df = (
        training_feature_df
        .filter(F.col(config["training_filter_col"]) == 1)
        .filter(F.col(label_col).isNotNull())
    )

    scoring_base_df = (
        training_feature_df
        .filter(F.col(config["scoring_filter_col"]) == 1)
    )

    print("Probability training rows:", training_base_df.count())
    print("Probability scoring rows:", scoring_base_df.count())

    train_df, test_df, cutoff_date = build_time_split(training_base_df, config)

    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=config["model_name"]):
        train_count = train_df.count()
        test_count = test_df.count()
        scoring_count = scoring_base_df.count()

        mlflow.log_param("branch_name", config["branch_name"])
        mlflow.log_param("model_name", config["model_name"])
        mlflow.log_param("model_type", "daily_hierarchical_mean_deuplifted_probability_target")
        mlflow.log_param("label_col", label_col)
        mlflow.log_param("training_rows", train_count)
        mlflow.log_param("test_rows", test_count)
        mlflow.log_param("scoring_rows", scoring_count)
        mlflow.log_param("cutoff_date", cutoff_date)
        mlflow.log_param("daily_grain", "product_store_day")
        mlflow.log_param("probability_floor", PROBABILITY_FLOOR)
        mlflow.log_param("probability_ceiling", PROBABILITY_CEILING)

        test_predictions_raw, global_prediction = add_hierarchical_mean_prediction(
            train_df=train_df,
            score_df=test_df,
            label_col=label_col,
            prediction_col=config["prediction_col"],
            config=config,
        )

        test_predictions = (
            test_predictions_raw
            .withColumn(
                "predicted_deuplifted_probability",
                clip_probability_expr(F.col(config["prediction_col"]))
            )
        )

        metrics_eval = compute_regression_metrics(
            scored_df=test_predictions,
            label_col=label_col,
            prediction_col="predicted_deuplifted_probability",
        )

        rmse = metrics_eval["rmse"]
        mae = metrics_eval["mae"]
        r2 = metrics_eval["r2"]

        mlflow.log_param("global_prediction", global_prediction)

        if rmse is not None:
            mlflow.log_metric("rmse", rmse)
        if mae is not None:
            mlflow.log_metric("mae", mae)
        if r2 is not None:
            mlflow.log_metric("r2", r2)

        scored_df, _ = add_hierarchical_mean_prediction(
            train_df=train_df,
            score_df=scoring_base_df,
            label_col=label_col,
            prediction_col=config["prediction_col"],
            config=config,
        )

    output_cols = get_existing_cols(scored_df, config["identity_cols"])

    probability_output_df = (
        scored_df
        .withColumn(
            "predicted_deuplifted_probability_raw",
            F.col(config["prediction_col"])
        )
        .withColumn(
            "predicted_deuplifted_probability",
            clip_probability_expr(F.col("predicted_deuplifted_probability_raw"))
        )
        .withColumn(
            "predicted_deuplifted_probability_logit",
            logit_expr(F.col("predicted_deuplifted_probability"))
        )
        .withColumn("prediction_branch", F.lit("probability"))
        .withColumn("prediction_model_name", F.lit(config["model_name"]))
        .withColumn("prediction_model_type", F.lit("daily_hierarchical_mean_deuplifted_probability_target"))
        .withColumn("model_created_at", F.current_timestamp())
        .select(
            *output_cols,
            F.col("probability_target").alias("actual_probability_target"),
            F.col("probability_logit_target").alias("actual_probability_logit"),
            F.col("probability_current_discount_effect"),
            F.col("probability_deuplifted_logit"),
            F.col("probability_deuplifted_target"),
            F.col("predicted_deuplifted_probability_raw"),
            F.col("predicted_deuplifted_probability"),
            F.col("predicted_deuplifted_probability_logit"),
            "prediction_branch",
            "prediction_model_name",
            "prediction_model_type",
            "model_created_at"
        )
    )

    (
        probability_output_df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(config["output_table"])
    )

    probability_output_df.createOrReplaceTempView(config["output_view"])

    metrics = {
        "model_type": "daily_hierarchical_mean_deuplifted_probability_target",
        "target_col": label_col,
        "training_filter_col": config["training_filter_col"],
        "scoring_filter_col": config["scoring_filter_col"],
        "train_rows": train_count,
        "test_rows": test_count,
        "scoring_rows": scoring_count,
        "cutoff_date": cutoff_date,
        "global_prediction": global_prediction,
        "probability_floor": PROBABILITY_FLOOR,
        "probability_ceiling": PROBABILITY_CEILING,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "output_table": config["output_table"],
    }

    print("Daily probability predictive model completed.")
    print("Probability output table:", config["output_table"])
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    cleanup_memory()

    return metric_rows(
        config["branch_name"],
        config["model_name"],
        metrics
    )


# ============================================================
# 12. Add scenario uplift back
# ============================================================

def create_predictive_scenario_output(sales_coefs: List[float], probability_coefs: List[float]):
    print("==================================================")
    print("Creating DAILY predictive scenario output")
    print("==================================================")

    require_table(SALES_PREDICTION_OUTPUT_TABLE)
    require_table(PROBABILITY_PREDICTION_OUTPUT_TABLE)
    require_table(SALES_DISCOUNT_COEFFICIENT_TABLE)

    sales_df = spark.table(SALES_PREDICTION_OUTPUT_TABLE)
    probability_df = spark.table(PROBABILITY_PREDICTION_OUTPUT_TABLE)

    sales_coef_df = load_sales_discount_coefficient_df()

    # Keep the configured discount ladder, but always include 0% as the
    # technical PE baseline. The current discount is added dynamically below,
    # even when it is not present in this configured list.
    configured_scenarios = sorted(
        {
            0.0,
            *[
                float(value)
                for value in get_common_config()["scenario_discounts"]
            ],
        }
    )

    configured_scenario_array = F.array(
        *[
            F.lit(float(value)).cast("double")
            for value in configured_scenarios
        ]
    )

    join_cols = [
        "date",
        "wm_yr_wk",
        "pe_article",
        "pe_store_group",
        "pe_article_store_group",
    ]

    def add_dynamic_scenarios(input_df):
        """
        Add three scenario categories without changing the downstream flow:

        TECHNICAL_BASELINE:
            0% discount retained for PE calculations.

        CURRENT_BASELINE:
            The discount currently active for the product-store-date. It is
            always added dynamically, even when it is not in the configured
            scenario list.

        CANDIDATE:
            Configured discounts strictly greater than the current discount
            and less than or equal to MAX_SCENARIO_DISCOUNT.

        When current discount is 0%, the same row is both the PE technical
        baseline and the business current baseline. scenario_type is marked as
        CURRENT_BASELINE, while both flag columns remain available.
        """

        current_discount_expr = (
            F.when(F.col("discount").isNull(), F.lit(0.0))
            .when(F.col("discount") > 1.0, F.col("discount") / F.lit(100.0))
            .otherwise(F.col("discount").cast("double"))
        )

        return (
            input_df
            .withColumn(
                "current_discount_normalized",
                F.greatest(
                    F.lit(0.0),
                    F.least(current_discount_expr, F.lit(0.999999)),
                ),
            )
            .withColumn(
                "_scenario_values",
                F.array_distinct(
                    F.concat(
                        configured_scenario_array,
                        F.array(F.col("current_discount_normalized")),
                    )
                ),
            )
            .withColumn(
                "scenario_discount",
                F.explode(F.col("_scenario_values")),
            )
            .drop("_scenario_values")
            .withColumn(
                "is_technical_baseline",
                F.when(
                    F.abs(F.col("scenario_discount")) <= F.lit(0.0005),
                    F.lit(1),
                ).otherwise(F.lit(0)),
            )
            .withColumn(
                "is_current_baseline",
                F.when(
                    F.abs(
                        F.col("scenario_discount")
                        - F.col("current_discount_normalized")
                    ) <= F.lit(0.0005),
                    F.lit(1),
                ).otherwise(F.lit(0)),
            )
            .withColumn(
                "is_candidate_scenario",
                F.when(
                    F.col("scenario_discount")
                    > F.col("current_discount_normalized") + F.lit(0.0005),
                    F.lit(1),
                ).otherwise(F.lit(0)),
            )
            .withColumn(
                "scenario_type",
                F.when(
                    F.col("is_current_baseline") == 1,
                    F.lit("CURRENT_BASELINE"),
                )
                .when(
                    F.col("is_technical_baseline") == 1,
                    F.lit("TECHNICAL_BASELINE"),
                )
                .when(
                    F.col("is_candidate_scenario") == 1,
                    F.lit("CANDIDATE"),
                )
                .otherwise(F.lit("BELOW_CURRENT")),
            )
            .filter(
                (F.col("is_technical_baseline") == 1)
                | (F.col("is_current_baseline") == 1)
                | (
                    (F.col("is_candidate_scenario") == 1)
                    & (
                        F.col("scenario_discount")
                        <= F.lit(MAX_SCENARIO_DISCOUNT)
                    )
                )
            )
        )

    drop_existing_coef_cols = [
        col_name
        for col_name in (
            COEFFICIENT_COLS + [
                "coefficient_level",
                "coefficient_source",
                "model_status",
                "random_effect_applied",
                "coefficient_history_rows",
                "coefficient_discount_variation_count",
                "global_discount_effect_coefficient_1",
                "random_discount_effect_coefficient_1",
                "estimated_elasticity",
                "elasticity_source",
                "elasticity_status",
            ]
        )
        if col_name in sales_df.columns
    ]

    sales_df_clean = sales_df.drop(*drop_existing_coef_cols)

    sales_scenario_df = (
        add_dynamic_scenarios(
            sales_df_clean.join(
                sales_coef_df,
                on="pe_article_store_group",
                how="left",
            )
        )
        # Determine the regular/reference price only once. This prevents the
        # current discount or recommended discount from being applied twice.
        .withColumn(
            "_scenario_regular_price",
            F.when(
                F.col("pe_average_zone_retail_price").isNotNull()
                & (F.col("pe_average_zone_retail_price") > 0),
                F.col("pe_average_zone_retail_price"),
            )
            .when(
                F.col("pe_actual_retail_price").isNotNull()
                & (F.col("pe_actual_retail_price") > 0),
                F.col("pe_actual_retail_price"),
            )
            .when(
                F.col("pe_unit_price").isNotNull()
                & (F.col("pe_unit_price") > 0)
                & (F.col("current_discount_normalized") < 1.0),
                F.col("pe_unit_price")
                / (F.lit(1.0) - F.col("current_discount_normalized")),
            )
            .otherwise(F.col("pe_unit_price")),
        )
        .withColumn(
            "scenario_unit_price",
            F.col("_scenario_regular_price")
            * (F.lit(1.0) - F.col("scenario_discount")),
        )
        .withColumn(
            "scenario_unit_price",
            F.when(
                F.col("scenario_unit_price") < 0,
                F.lit(0.0),
            ).otherwise(F.col("scenario_unit_price")),
        )
        .withColumn(
            "scenario_price_ratio",
            F.when(
                F.col("_scenario_regular_price").isNotNull()
                & (F.col("_scenario_regular_price") > 0)
                & F.col("scenario_unit_price").isNotNull(),
                F.col("scenario_unit_price")
                / F.col("_scenario_regular_price"),
            ).otherwise(F.lit(1.0) - F.col("scenario_discount")),
        )
        .withColumn(
            "scenario_price_ratio",
            F.when(
                F.col("scenario_price_ratio") <= 0,
                F.lit(1.0),
            ).otherwise(F.col("scenario_price_ratio")),
        )
        .withColumn(
            "sales_scenario_discount_effect_raw",
            row_log_price_elasticity_expr("scenario_price_ratio"),
        )
        .withColumn(
            "sales_scenario_discount_effect",
            F.least(
                F.greatest(
                    F.col("sales_scenario_discount_effect_raw"),
                    F.lit(0.0),
                ),
                F.lit(MAX_SALES_SCENARIO_LOG_UPLIFT),
            ),
        )
        .withColumn(
            "scenario_base_log_quantity",
            F.coalesce(
                F.col("predicted_deuplifted_log_quantity_corrected"),
                F.col("predicted_deuplifted_log_quantity"),
            ),
        )
        .withColumn(
            "predictive_scenario_log_quantity",
            F.col("scenario_base_log_quantity")
            + F.col("sales_scenario_discount_effect"),
        )
        .withColumn(
            "predictive_scenario_quantity",
            F.exp(F.col("predictive_scenario_log_quantity")),
        )
    )

    probability_scenario_df = (
        add_dynamic_scenarios(probability_df)
        .withColumn(
            "probability_scenario_discount_effect",
            polynomial_expr("scenario_discount", probability_coefs),
        )
        .withColumn(
            "predictive_scenario_probability_logit",
            F.col("predicted_deuplifted_probability_logit")
            + F.col("probability_scenario_discount_effect"),
        )
        .withColumn(
            "predictive_scenario_probability",
            sigmoid_expr(F.col("predictive_scenario_probability_logit")),
        )
        .withColumn(
            "predictive_scenario_probability",
            clip_probability_expr(F.col("predictive_scenario_probability")),
        )
    )

    # Rule 30 - Minimum History Rule:
    sales_scenario_df = ensure_column(
        sales_scenario_df,
        "history_days",
        None,
        "double",
    )

    # Rule 31 - Price Variation Rule:
    sales_scenario_df = ensure_column(
        sales_scenario_df,
        "price_variation",
        None,
        "double",
    )

    # Rule 32 - Discount Variation Rule:
    sales_scenario_df = ensure_column(
        sales_scenario_df,
        "discount_variation",
        None,
        "double",
    )

    # Rule 33 - Low Confidence Prediction Rule:
    sales_scenario_df = ensure_column(
        sales_scenario_df,
        "prediction_confidence",
        None,
        "double",
    )

    combined_df = (
        sales_scenario_df.alias("s")
        .join(
            probability_scenario_df.alias("p"),
            on=join_cols + ["scenario_discount"],
            how="inner",
        )
        .select(
            F.col("s.base_data_grain").alias("base_data_grain"),

            F.col("date"),
            F.col("wm_yr_wk"),
            F.col("pe_article"),
            F.col("pe_store_group"),
            F.col("pe_article_store_group"),

            F.col("scenario_discount"),
            F.col("s.scenario_type").alias("scenario_type"),
            F.col("s.is_technical_baseline").alias("is_technical_baseline"),
            F.col("s.is_current_baseline").alias("is_current_baseline"),
            F.col("s.is_candidate_scenario").alias("is_candidate_scenario"),

            F.col("s.pe_quantity").alias("actual_quantity"),
            F.col("s.pe_unit_price").alias("pe_unit_price"),
            F.col("s.pe_actual_retail_price").alias("pe_actual_retail_price"),
            F.col("s.pe_average_zone_retail_price").alias("pe_average_zone_retail_price"),
            F.col("s.current_discount_normalized").alias("current_discount"),
            F.col("s.scenario_unit_price").alias("scenario_unit_price"),

            F.col("s.predicted_deuplifted_log_quantity").alias("predicted_deuplifted_log_quantity"),
            F.col("s.predicted_deuplifted_quantity").alias("predicted_deuplifted_quantity"),

            F.col("s.use_sales_arma_correction").alias("use_sales_arma_correction"),
            F.col("s.use_sales_residual_correction").alias("use_sales_residual_correction"),

            F.col("s.arma_history_points").alias("arma_history_points"),
            F.col("s.arma_phi").alias("arma_phi"),
            F.col("s.arma_residual_correction").alias("arma_residual_correction"),
            F.col("s.arma_last_date").alias("arma_last_date"),
            F.col("s.arma_last_week").alias("arma_last_week"),

            F.col("s.sales_residual_correction").alias("sales_residual_correction"),
            F.col("s.sales_residual_points").alias("sales_residual_points"),
            F.col("s.sales_residual_lookback_days").alias("sales_residual_lookback_days"),
            F.col("s.sales_residual_lookback_start_date").alias("sales_residual_lookback_start_date"),

            F.col("s.predicted_deuplifted_log_quantity_corrected").alias("predicted_deuplifted_log_quantity_corrected"),
            F.col("s.predicted_deuplifted_quantity_corrected").alias("predicted_deuplifted_quantity_corrected"),
            F.col("s.scenario_base_log_quantity").alias("scenario_base_log_quantity"),

            F.col("s.sales_scenario_discount_effect_raw").alias("sales_scenario_discount_effect_raw"),
            F.col("s.sales_scenario_discount_effect").alias("sales_scenario_discount_effect"),

            F.col("s.discount_effect_coefficient_1").alias("discount_effect_coefficient_1"),
            F.col("s.discount_effect_coefficient_2").alias("discount_effect_coefficient_2"),
            F.col("s.discount_effect_coefficient_3").alias("discount_effect_coefficient_3"),
            F.col("s.discount_effect_coefficient_4").alias("discount_effect_coefficient_4"),
            F.col("s.coefficient_level").alias("coefficient_level"),
            F.col("s.coefficient_source").alias("coefficient_source"),
            F.col("s.estimated_elasticity").alias("estimated_elasticity"),
            F.col("s.elasticity_source").alias("elasticity_source"),
            F.col("s.elasticity_status").alias("elasticity_status"),
            F.col("s.scenario_price_ratio").alias("scenario_price_ratio"),
            F.col("s.model_status").alias("coefficient_model_status"),
            F.col("s.random_effect_applied").alias("random_effect_applied"),
            F.col("s.coefficient_history_rows").alias("coefficient_history_rows"),
            F.col("s.coefficient_discount_variation_count").alias("coefficient_discount_variation_count"),

            F.col("s.predictive_scenario_log_quantity").alias("predictive_scenario_log_quantity"),
            F.col("s.predictive_scenario_quantity").alias("predictive_scenario_quantity"),

            F.col("p.predicted_deuplifted_probability_logit").alias("predicted_deuplifted_probability_logit"),
            F.col("p.predicted_deuplifted_probability").alias("predicted_deuplifted_probability"),
            F.col("p.probability_scenario_discount_effect").alias("probability_scenario_discount_effect"),
            F.col("p.predictive_scenario_probability_logit").alias("predictive_scenario_probability_logit"),
            F.col("p.predictive_scenario_probability").alias("predictive_scenario_probability"),

            (
                F.col("s.predictive_scenario_quantity")
                * F.col("p.predictive_scenario_probability")
            ).alias("predictive_expected_quantity"),

            (
                F.col("s.predictive_scenario_quantity")
                * F.col("p.predictive_scenario_probability")
                * F.col("s.scenario_unit_price")
            ).alias("predictive_expected_revenue"),

            F.col("s.valid_for_price_elasticity").alias("valid_for_price_elasticity"),
            F.col("s.valid_for_mdo_input").alias("valid_for_mdo_input"),
            F.col("s.history_days").alias("history_days"),
            F.col("s.price_variation").alias("price_variation"),
            F.col("s.discount_variation").alias("discount_variation"),
            F.col("s.prediction_confidence").alias("prediction_confidence"),

            F.col("s.pe_store_stock_quantity").alias("pe_store_stock_quantity"),
            F.col("s.pe_inventory_onhand_quantity").alias("pe_inventory_onhand_quantity"),
            F.col("s.stock_available_flag").alias("stock_available_flag"),
            F.col("s.inventory_available_flag").alias("inventory_available_flag"),

            F.col("s.pe_category").alias("pe_category"),
            F.col("s.pe_product_type").alias("pe_product_type"),
            F.col("s.pe_product_division").alias("pe_product_division"),
            F.col("s.pe_country").alias("pe_country"),

            F.current_timestamp().alias("model_created_at"),
        )
        .dropDuplicates(
            [
                "date",
                "pe_article",
                "pe_store_group",
                "scenario_discount",
            ]
        )
    )

    (
        combined_df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(PREDICTIVE_SCENARIO_OUTPUT_TABLE)
    )

    combined_df.createOrReplaceTempView(PREDICTIVE_SCENARIO_OUTPUT_VIEW)

    print("Daily predictive scenario output saved:", PREDICTIVE_SCENARIO_OUTPUT_TABLE)

    print("Validation: dynamic scenario types")
    display(
        spark.sql(f"""
            SELECT
                scenario_type,
                is_technical_baseline,
                is_current_baseline,
                is_candidate_scenario,
                COUNT(*) AS rows
            FROM {PREDICTIVE_SCENARIO_OUTPUT_TABLE}
            GROUP BY
                scenario_type,
                is_technical_baseline,
                is_current_baseline,
                is_candidate_scenario
            ORDER BY scenario_type
        """)
    )

    print("Validation: daily scenario grain check")
    display(
        spark.sql(f"""
            SELECT
                COUNT(*) AS duplicate_keys
            FROM (
                SELECT
                    date,
                    pe_article,
                    pe_store_group,
                    scenario_discount,
                    COUNT(*) AS rows_per_key
                FROM {PREDICTIVE_SCENARIO_OUTPUT_TABLE}
                GROUP BY
                    date,
                    pe_article,
                    pe_store_group,
                    scenario_discount
                HAVING COUNT(*) > 1
            )
        """)
    )

    cleanup_memory()

    return None


# ============================================================
# 13. Runner
# ============================================================

def run_predictive_models():
    print("==================================================")
    print("Running DAILY PE predictive model flow with optional ARMA residual correction")
    print("==================================================")

    training_table_name, sales_coefs, probability_coefs = build_predictive_training_features()

    sales_metric_rows = train_sales_predictive_model(
        training_table_name
    )

    cleanup_memory()

    probability_metric_rows = train_probability_predictive_model(
        training_table_name
    )

    cleanup_memory()

    create_predictive_scenario_output(
        sales_coefs=sales_coefs,
        probability_coefs=probability_coefs
    )

    cleanup_memory()

    all_metric_rows = sales_metric_rows + probability_metric_rows

    summary_df = spark.createDataFrame(
        all_metric_rows,
        ["branch_name", "model_name", "metric", "value"]
    )

    (
        summary_df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(PREDICTIVE_SUMMARY_TABLE)
    )

    print("Predictive model summary saved:", PREDICTIVE_SUMMARY_TABLE)

    print("Validation 1: predictive summary")
    display(
        spark.sql(f"""
            SELECT *
            FROM {PREDICTIVE_SUMMARY_TABLE}
            ORDER BY branch_name, model_name, metric
        """)
    )

    print("Validation 2: output row counts")
    display(
        spark.sql(f"""
            SELECT 'predictive_training_features' AS output_name, COUNT(*) AS rows
            FROM {PREDICTIVE_TRAINING_TABLE}

            UNION ALL

            SELECT 'sales_prediction_output' AS output_name, COUNT(*) AS rows
            FROM {SALES_PREDICTION_OUTPUT_TABLE}

            UNION ALL

            SELECT 'probability_prediction_output' AS output_name, COUNT(*) AS rows
            FROM {PROBABILITY_PREDICTION_OUTPUT_TABLE}

            UNION ALL

            SELECT 'predictive_scenario_output' AS output_name, COUNT(*) AS rows
            FROM {PREDICTIVE_SCENARIO_OUTPUT_TABLE}
        """)
    )

    print("Validation 3: ARMA residual correction decision")
    display(
        spark.sql(f"""
            SELECT
                COUNT(*) AS rows,
                MIN(use_sales_arma_correction) AS min_use_sales_arma_correction,
                MAX(use_sales_arma_correction) AS max_use_sales_arma_correction,
                AVG(use_sales_arma_correction) AS avg_use_sales_arma_correction,
                MIN(arma_residual_correction) AS min_arma_residual_correction,
                MAX(arma_residual_correction) AS max_arma_residual_correction,
                AVG(arma_residual_correction) AS avg_arma_residual_correction,
                AVG(arma_history_points) AS avg_arma_history_points,
                AVG(arma_phi) AS avg_arma_phi
            FROM {SALES_PREDICTION_OUTPUT_TABLE}
        """)
    )

    print("Validation 4: scenario probability and capped sales uplift distribution")
    display(
        spark.sql(f"""
            SELECT
                scenario_discount,
                MIN(predictive_scenario_probability) AS min_predictive_scenario_probability,
                MAX(predictive_scenario_probability) AS max_predictive_scenario_probability,
                AVG(predictive_scenario_probability) AS avg_predictive_scenario_probability,
                AVG(sales_scenario_discount_effect_raw) AS avg_sales_scenario_discount_effect_raw,
                AVG(sales_scenario_discount_effect) AS avg_sales_scenario_discount_effect_capped,
                AVG(discount_effect_coefficient_1) AS avg_discount_effect_coefficient_1,
                MIN(discount_effect_coefficient_1) AS min_discount_effect_coefficient_1,
                MAX(discount_effect_coefficient_1) AS max_discount_effect_coefficient_1,
                STDDEV(discount_effect_coefficient_1) AS stddev_discount_effect_coefficient_1,
                AVG(estimated_elasticity) AS avg_estimated_elasticity,
                MIN(estimated_elasticity) AS min_estimated_elasticity,
                MAX(estimated_elasticity) AS max_estimated_elasticity,
                AVG(use_sales_arma_correction) AS avg_use_sales_arma_correction,
                AVG(arma_residual_correction) AS avg_arma_residual_correction,
                AVG(predictive_scenario_quantity) AS avg_predictive_scenario_quantity,
                AVG(predictive_expected_quantity) AS avg_predictive_expected_quantity,
                AVG(predictive_expected_revenue) AS avg_predictive_expected_revenue
            FROM {PREDICTIVE_SCENARIO_OUTPUT_TABLE}
            GROUP BY scenario_discount
            ORDER BY scenario_discount
        """)
    )

    print("Validation 5: expected quantity ratio vs zero discount")
    display(
        spark.sql(f"""
            WITH scenario_avg AS (
                SELECT
                    scenario_discount,
                    AVG(predictive_expected_quantity) AS avg_expected_quantity
                FROM {PREDICTIVE_SCENARIO_OUTPUT_TABLE}
                GROUP BY scenario_discount
            )
            SELECT
                scenario_discount,
                avg_expected_quantity,
                avg_expected_quantity /
                    FIRST_VALUE(avg_expected_quantity) OVER (
                        ORDER BY scenario_discount
                    ) AS expected_quantity_ratio_vs_zero_discount
            FROM scenario_avg
            ORDER BY scenario_discount
        """)
    )

    print("Validation 6: product coefficient variation")
    display(
        spark.sql(f"""
            SELECT
                pe_article,
                COUNT(DISTINCT pe_article_store_group) AS product_store_groups,
                AVG(discount_effect_coefficient_1) AS avg_coef_1,
                MIN(discount_effect_coefficient_1) AS min_coef_1,
                MAX(discount_effect_coefficient_1) AS max_coef_1,
                STDDEV(discount_effect_coefficient_1) AS stddev_coef_1
            FROM {PREDICTIVE_SCENARIO_OUTPUT_TABLE}
            GROUP BY pe_article
            ORDER BY pe_article
        """)
    )

    print("Validation 7: exact daily query example")
    display(
        spark.sql(f"""
            SELECT
                date,
                pe_article,
                pe_store_group,
                scenario_discount,
                actual_quantity,
                predictive_scenario_quantity,
                predictive_scenario_probability,
                predictive_expected_quantity,
                predictive_expected_revenue,
                scenario_unit_price,
                scenario_price_ratio,
                estimated_elasticity,
                elasticity_source,
                elasticity_status,
                discount_effect_coefficient_1,
                coefficient_level,
                coefficient_source
            FROM {PREDICTIVE_SCENARIO_OUTPUT_TABLE}
            ORDER BY date, pe_store_group, pe_article, scenario_discount
            LIMIT 100
        """)
    )

    print("==================================================")
    print("DAILY PE predictive model flow completed")
    print("Predictive training table:", PREDICTIVE_TRAINING_TABLE)
    print("Sales output:", SALES_PREDICTION_OUTPUT_TABLE)
    print("Probability output:", PROBABILITY_PREDICTION_OUTPUT_TABLE)
    print("Scenario output:", PREDICTIVE_SCENARIO_OUTPUT_TABLE)
    print("Summary:", PREDICTIVE_SUMMARY_TABLE)
    print("==================================================")

    cleanup_memory()

    return {
        "predictive_training_table": PREDICTIVE_TRAINING_TABLE,
        "sales_output_table": SALES_PREDICTION_OUTPUT_TABLE,
        "probability_output_table": PROBABILITY_PREDICTION_OUTPUT_TABLE,
        "scenario_output_table": PREDICTIVE_SCENARIO_OUTPUT_TABLE,
        "summary_table": PREDICTIVE_SUMMARY_TABLE,
    }


# ============================================================
# 14. Execute
# ============================================================

if __name__ == "__main__":
    predictive_outputs = run_predictive_models()