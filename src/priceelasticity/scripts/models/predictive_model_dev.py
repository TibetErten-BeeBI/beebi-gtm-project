import sys
import gc
import math
from typing import Dict, List, Tuple

import mlflow
# ============================================================
# MLflow experiment setup for Databricks Jobs
# ============================================================

MLFLOW_EXPERIMENT_NAME = "/Users/vadali.tejasviram@beebi-consulting.com/PE_MDO_Model_Experiments"

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. Spark setup
# ============================================================

spark = SparkSession.builder.getOrCreate()

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
# 2. Table config
# ============================================================

FEATURE_TABLE = "workspace.default.pe_causal_features_dev"

SALES_MIXEDLINEAR_COUNTERFACTUAL_TABLE = "workspace.default.pe_sales_mixedlinear_counterfactual"
PROBABILITY_MIXEDLINEAR_COUNTERFACTUAL_TABLE = "workspace.default.pe_probability_mixedlinear_counterfactual"

PREDICTIVE_TRAINING_TABLE = "workspace.default.pe_predictive_training_features"

SALES_PREDICTION_OUTPUT_TABLE = "workspace.default.pe_sales_prediction_output"
PROBABILITY_PREDICTION_OUTPUT_TABLE = "workspace.default.pe_probability_prediction_output"
PREDICTIVE_SCENARIO_OUTPUT_TABLE = "workspace.default.pe_predictive_scenario_output"
PREDICTIVE_SUMMARY_TABLE = "workspace.default.pe_predictive_model_summary"

SALES_PREDICTION_OUTPUT_VIEW = "pe_sales_prediction_output_view"
PROBABILITY_PREDICTION_OUTPUT_VIEW = "pe_probability_prediction_output_view"
PREDICTIVE_SCENARIO_OUTPUT_VIEW = "pe_predictive_scenario_output_view"

RANDOM_SEED = 42

PROBABILITY_FLOOR = 0.01
PROBABILITY_CEILING = 0.99

# Business safety cap:
# Maximum allowed sales scenario uplift on log scale.
# exp(1.20) is about 3.32x max uplift from discount.
MAX_SALES_SCENARIO_LOG_UPLIFT = 1.20

# ARMA residual model settings.
# This follows old-code ideology:
# baseline prediction + ARMA residual correction.
ARMA_MIN_HISTORY_POINTS = 8
MAX_SALES_ARMA_CORRECTION = 0.75
ARMA_PHI_FLOOR = -0.80
ARMA_PHI_CEILING = 0.80

# Critical safety rule:
# Use ARMA correction only if it improves validation RMSE.
MIN_RMSE_IMPROVEMENT_TO_USE_ARMA = 0.0


# ============================================================
# 3. Memory cleanup
# ============================================================

def cleanup_memory():
    try:
        gc.collect()
    except Exception as exc:
        print("Python garbage collection skipped:", exc)


# ============================================================
# 4. Config
# ============================================================

def get_common_config() -> Dict:
    return {
        "date_col": "date",
        "week_col": "wm_yr_wk",
        "article_col": "pe_article",
        "store_col": "pe_store_group",
        "group_col": "pe_article_store_group",

        "scenario_discounts": [
            0.00,
            0.05,
            0.075,
            0.10,
            0.15,
            0.20,
        ],

        "identity_cols": [
            "date",
            "wm_yr_wk",
            "pe_article",
            "pe_store_group",
            "pe_article_store_group",
            "pe_quantity",
            "pe_unit_price",
            "discount",
            "valid_price_flag",
            "valid_for_price_elasticity",
            "valid_for_mdo_input",
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
        "model_name": "sales_deuplifted_predictive_model",

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
        "model_name": "probability_deuplifted_predictive_model",

        # Train probability on bounded probability target, not logit.
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
# 5. Helpers
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


def polynomial_expr(discount_col: str, coefs: List[float]):
    expr = F.lit(0.0)

    for idx, coef in enumerate(coefs, start=1):
        expr = expr + F.lit(float(coef)) * F.pow(
            F.col(discount_col).cast("double"),
            F.lit(idx)
        )

    return expr


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


def load_discount_coefficients(counterfactual_table: str, label: str) -> List[float]:
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
    week_col = config["week_col"]
    split_ratio = config["train_test_split_ratio"]

    weeks = [
        row[week_col]
        for row in (
            df
            .select(week_col)
            .where(F.col(week_col).isNotNull())
            .distinct()
            .orderBy(week_col)
            .collect()
        )
    ]

    if len(weeks) < 3:
        print("Not enough weeks for time split. Using random split.")
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
        return train_df, test_df, None

    cutoff_index = max(1, int(len(weeks) * split_ratio) - 1)
    cutoff_week = weeks[cutoff_index]

    train_df = df.filter(F.col(week_col) <= F.lit(cutoff_week))
    test_df = df.filter(F.col(week_col) > F.lit(cutoff_week))

    if test_df.count() == 0:
        print("Time split produced empty test set. Using random split.")
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
        return train_df, test_df, None

    print("Time split cutoff week:", cutoff_week)

    return train_df, test_df, cutoff_week


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
    Build residual training data for ARMA.

    residual = actual de-uplifted sales - baseline predicted de-uplifted sales
    """

    group_col = config["group_col"]
    week_col = config["week_col"]

    train_predictions, _ = add_hierarchical_mean_prediction(
        train_df=train_df,
        score_df=train_df,
        label_col=label_col,
        prediction_col=prediction_col,
        config=config,
    )

    residual_df = (
        train_predictions
        .withColumn(
            "sales_residual",
            F.col(label_col) - F.col(prediction_col)
        )
        .select(
            F.col(group_col),
            F.col(week_col),
            F.col("sales_residual"),
        )
        .filter(F.col("sales_residual").isNotNull())
    )

    return residual_df


def apply_sales_arma_correction(
    scored_df,
    arma_model,
    prediction_col: str,
):
    """
    Apply ARMA residual correction to baseline sales prediction.
    """

    scored_with_arma_df = (
        arma_model
        .transform(
            score_df=scored_df,
            output_col="arma_residual_correction",
        )
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
        .withColumn("arma_last_week", F.lit(None).cast("long"))

        .withColumn("sales_residual_correction_raw", F.lit(0.0))
        .withColumn("sales_residual_correction", F.lit(0.0))
        .withColumn("sales_residual_points", F.lit(0))
        .withColumn("sales_residual_lookback_weeks", F.lit(ARMA_MIN_HISTORY_POINTS))
        .withColumn("sales_residual_lookback_start_week", F.lit(None).cast("long"))

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
# 6. Build de-uplifted predictive training features
# ============================================================

def build_predictive_training_features():
    print("==================================================")
    print("Building de-uplifted predictive training features")
    print("==================================================")

    require_table(FEATURE_TABLE)
    require_table(SALES_MIXEDLINEAR_COUNTERFACTUAL_TABLE)
    require_table(PROBABILITY_MIXEDLINEAR_COUNTERFACTUAL_TABLE)

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

    sales_coefs = load_discount_coefficients(
        SALES_MIXEDLINEAR_COUNTERFACTUAL_TABLE,
        "sales mixed-linear"
    )

    probability_coefs = load_discount_coefficients(
        PROBABILITY_MIXEDLINEAR_COUNTERFACTUAL_TABLE,
        "probability mixed-linear"
    )

    # Business safety rule:
    # If probability discount effect is negative, do not use it.
    # Discount impact will still be handled by the sales quantity model.
    if probability_coefs[0] < 0:
        print("WARNING: Negative probability discount coefficient found.")
        print("Original probability coefficients:", probability_coefs)
        print("Setting probability discount coefficients to zero for predictive/MDO safety.")
        probability_coefs = [0.0, 0.0, 0.0, 0.0]

    predictive_df = (
        feature_df
        .withColumn(
            "sales_current_discount_effect",
            polynomial_expr("discount", sales_coefs)
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
        .withColumn("sales_uplift_source_table", F.lit(SALES_MIXEDLINEAR_COUNTERFACTUAL_TABLE))
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
                COUNT(*) AS rows,
                COUNT(DISTINCT pe_article) AS products,
                COUNT(DISTINCT pe_store_group) AS stores,
                COUNT(DISTINCT pe_article_store_group) AS product_store_groups,
                COUNT(DISTINCT wm_yr_wk) AS weeks,
                SUM(valid_for_sales_training) AS sales_training_rows,
                SUM(valid_for_probability_training_with_price) AS probability_training_rows
            FROM {PREDICTIVE_TRAINING_TABLE}
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
# 7. Train sales predictive model
# ============================================================

def train_sales_predictive_model(training_table_name: str):
    config = get_sales_predictive_config()

    print("==================================================")
    print("Training sales predictive model with optional ARMA residual correction")
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

    train_df, test_df, cutoff_week = build_time_split(training_base_df, config)

    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=config["model_name"]):
        train_count = train_df.count()
        test_count = test_df.count()
        scoring_count = scoring_base_df.count()

        mlflow.log_param("branch_name", config["branch_name"])
        mlflow.log_param("model_name", config["model_name"])
        mlflow.log_param("model_type", "hierarchical_mean_deuplifted_with_optional_arma_residual")
        mlflow.log_param("label_col", label_col)
        mlflow.log_param("training_rows", train_count)
        mlflow.log_param("test_rows", test_count)
        mlflow.log_param("scoring_rows", scoring_count)
        mlflow.log_param("cutoff_week", cutoff_week)

        mlflow.log_param("arma_available", ARMA_AVAILABLE)
        mlflow.log_param("arma_import_error", ARMA_IMPORT_ERROR)
        mlflow.log_param("arma_min_history_points", ARMA_MIN_HISTORY_POINTS)
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

                arma_model = ARMAResidualModel(
                    group_col=config["group_col"],
                    week_col=config["week_col"],
                    residual_col="sales_residual",
                    min_history_points=ARMA_MIN_HISTORY_POINTS,
                    max_abs_correction=MAX_SALES_ARMA_CORRECTION,
                    phi_floor=ARMA_PHI_FLOOR,
                    phi_ceiling=ARMA_PHI_CEILING,
                )

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
        .withColumn("prediction_model_type", F.lit("hierarchical_mean_deuplifted_with_optional_arma_residual"))
        .withColumn("model_created_at", F.current_timestamp())
        .select(
            *output_cols,
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
            F.col("arma_last_week"),

            F.col("sales_residual_correction_raw"),
            F.col("sales_residual_correction"),
            F.col("sales_residual_points"),
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
        "model_type": "hierarchical_mean_deuplifted_with_optional_arma_residual",
        "target_col": label_col,
        "training_filter_col": config["training_filter_col"],
        "scoring_filter_col": config["scoring_filter_col"],
        "train_rows": train_count,
        "test_rows": test_count,
        "scoring_rows": scoring_count,
        "cutoff_week": cutoff_week,
        "global_prediction": global_prediction,

        "arma_available": ARMA_AVAILABLE,
        "arma_import_error": ARMA_IMPORT_ERROR,
        "arma_failure_reason": arma_failure_reason,
        "arma_min_history_points": ARMA_MIN_HISTORY_POINTS,
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

        # Compatibility names.
        "rmse_corrected_candidate": rmse_arma,
        "mae_corrected_candidate": mae_arma,
        "r2_corrected_candidate": r2_arma,
        "rmse_improvement_from_residual": arma_rmse_improvement,

        "rmse": final_rmse,
        "mae": final_mae,
        "r2": final_r2,

        "output_table": config["output_table"],
    }

    print("Sales predictive model completed.")
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
# 8. Train probability predictive model
# ============================================================

def train_probability_predictive_model(training_table_name: str):
    config = get_probability_predictive_config()

    print("==================================================")
    print("Training probability predictive model with Spark-only hierarchical means")
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

    train_df, test_df, cutoff_week = build_time_split(training_base_df, config)

    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=config["model_name"]):
        train_count = train_df.count()
        test_count = test_df.count()
        scoring_count = scoring_base_df.count()

        mlflow.log_param("branch_name", config["branch_name"])
        mlflow.log_param("model_name", config["model_name"])
        mlflow.log_param("model_type", "hierarchical_mean_deuplifted_probability_target")
        mlflow.log_param("label_col", label_col)
        mlflow.log_param("training_rows", train_count)
        mlflow.log_param("test_rows", test_count)
        mlflow.log_param("scoring_rows", scoring_count)
        mlflow.log_param("cutoff_week", cutoff_week)
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
        .withColumn("prediction_model_type", F.lit("hierarchical_mean_deuplifted_probability_target"))
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
        "model_type": "hierarchical_mean_deuplifted_probability_target",
        "target_col": label_col,
        "training_filter_col": config["training_filter_col"],
        "scoring_filter_col": config["scoring_filter_col"],
        "train_rows": train_count,
        "test_rows": test_count,
        "scoring_rows": scoring_count,
        "cutoff_week": cutoff_week,
        "global_prediction": global_prediction,
        "probability_floor": PROBABILITY_FLOOR,
        "probability_ceiling": PROBABILITY_CEILING,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "output_table": config["output_table"],
    }

    print("Probability predictive model completed.")
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
# 9. Add scenario uplift back
# ============================================================

def create_predictive_scenario_output(sales_coefs: List[float], probability_coefs: List[float]):
    print("==================================================")
    print("Creating predictive scenario output")
    print("==================================================")

    require_table(SALES_PREDICTION_OUTPUT_TABLE)
    require_table(PROBABILITY_PREDICTION_OUTPUT_TABLE)

    sales_df = spark.table(SALES_PREDICTION_OUTPUT_TABLE)
    probability_df = spark.table(PROBABILITY_PREDICTION_OUTPUT_TABLE)

    scenario_df = create_scenario_df(get_common_config()["scenario_discounts"])

    join_cols = [
        "date",
        "wm_yr_wk",
        "pe_article",
        "pe_store_group",
        "pe_article_store_group",
    ]

    sales_scenario_df = (
        sales_df
        .crossJoin(scenario_df)

        .withColumn(
            "sales_scenario_discount_effect_raw",
            polynomial_expr("scenario_discount", sales_coefs)
        )
        .withColumn(
            "sales_scenario_discount_effect",
            F.least(
                F.greatest(
                    F.col("sales_scenario_discount_effect_raw"),
                    F.lit(0.0)
                ),
                F.lit(MAX_SALES_SCENARIO_LOG_UPLIFT)
            )
        )
        .withColumn(
            "scenario_base_log_quantity",
            F.coalesce(
                F.col("predicted_deuplifted_log_quantity_corrected"),
                F.col("predicted_deuplifted_log_quantity")
            )
        )
        .withColumn(
            "predictive_scenario_log_quantity",
            F.col("scenario_base_log_quantity") + F.col("sales_scenario_discount_effect")
        )
        .withColumn(
            "predictive_scenario_quantity",
            F.exp(F.col("predictive_scenario_log_quantity"))
        )
    )

    probability_scenario_df = (
        probability_df
        .crossJoin(scenario_df)
        .withColumn(
            "probability_scenario_discount_effect",
            polynomial_expr("scenario_discount", probability_coefs)
        )
        .withColumn(
            "predictive_scenario_probability_logit",
            F.col("predicted_deuplifted_probability_logit") + F.col("probability_scenario_discount_effect")
        )
        .withColumn(
            "predictive_scenario_probability",
            sigmoid_expr(F.col("predictive_scenario_probability_logit"))
        )
    )

    combined_df = (
        sales_scenario_df.alias("s")
        .join(
            probability_scenario_df.alias("p"),
            on=join_cols + ["scenario_discount"],
            how="inner"
        )
        .select(
            *[F.col(col_name) for col_name in join_cols],
            F.col("scenario_discount"),

            F.col("s.pe_quantity").alias("actual_quantity"),
            F.col("s.pe_unit_price").alias("pe_unit_price"),
            F.col("s.discount").alias("current_discount"),

            F.col("s.predicted_deuplifted_log_quantity").alias("predicted_deuplifted_log_quantity"),
            F.col("s.predicted_deuplifted_quantity").alias("predicted_deuplifted_quantity"),

            F.col("s.use_sales_arma_correction").alias("use_sales_arma_correction"),
            F.col("s.use_sales_residual_correction").alias("use_sales_residual_correction"),

            F.col("s.arma_history_points").alias("arma_history_points"),
            F.col("s.arma_phi").alias("arma_phi"),
            F.col("s.arma_residual_correction").alias("arma_residual_correction"),

            F.col("s.sales_residual_correction").alias("sales_residual_correction"),
            F.col("s.sales_residual_points").alias("sales_residual_points"),
            F.col("s.predicted_deuplifted_log_quantity_corrected").alias("predicted_deuplifted_log_quantity_corrected"),
            F.col("s.predicted_deuplifted_quantity_corrected").alias("predicted_deuplifted_quantity_corrected"),
            F.col("s.scenario_base_log_quantity").alias("scenario_base_log_quantity"),

            F.col("s.sales_scenario_discount_effect_raw").alias("sales_scenario_discount_effect_raw"),
            F.col("s.sales_scenario_discount_effect").alias("sales_scenario_discount_effect"),
            F.col("s.predictive_scenario_log_quantity").alias("predictive_scenario_log_quantity"),
            F.col("s.predictive_scenario_quantity").alias("predictive_scenario_quantity"),

            F.col("p.predicted_deuplifted_probability_logit").alias("predicted_deuplifted_probability_logit"),
            F.col("p.predicted_deuplifted_probability").alias("predicted_deuplifted_probability"),
            F.col("p.probability_scenario_discount_effect").alias("probability_scenario_discount_effect"),
            F.col("p.predictive_scenario_probability_logit").alias("predictive_scenario_probability_logit"),
            F.col("p.predictive_scenario_probability").alias("predictive_scenario_probability"),

            (
                F.col("s.predictive_scenario_quantity") *
                F.col("p.predictive_scenario_probability")
            ).alias("predictive_expected_quantity"),

            F.col("s.valid_for_price_elasticity").alias("valid_for_price_elasticity"),
            F.col("s.valid_for_mdo_input").alias("valid_for_mdo_input"),

            F.col("s.pe_category").alias("pe_category"),
            F.col("s.pe_product_type").alias("pe_product_type"),
            F.col("s.pe_product_division").alias("pe_product_division"),
            F.col("s.pe_country").alias("pe_country"),

            F.current_timestamp().alias("model_created_at")
        )
    )

    (
        combined_df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(PREDICTIVE_SCENARIO_OUTPUT_TABLE)
    )

    combined_df.createOrReplaceTempView(PREDICTIVE_SCENARIO_OUTPUT_VIEW)

    print("Predictive scenario output saved:", PREDICTIVE_SCENARIO_OUTPUT_TABLE)

    cleanup_memory()

    return None


# ============================================================
# 10. Runner
# ============================================================

def run_predictive_models():
    print("==================================================")
    print("Running PE predictive model flow with optional ARMA residual correction")
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
                AVG(use_sales_arma_correction) AS avg_use_sales_arma_correction,
                AVG(arma_residual_correction) AS avg_arma_residual_correction,
                AVG(predictive_scenario_quantity) AS avg_predictive_scenario_quantity,
                AVG(predictive_expected_quantity) AS avg_predictive_expected_quantity
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

    print("==================================================")
    print("PE predictive model flow completed")
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
# 11. Execute
# ============================================================
if __name__ == "__main__":
    predictive_outputs = run_predictive_models()