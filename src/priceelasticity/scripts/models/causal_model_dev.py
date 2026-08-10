import sys
import os
import importlib.util
import datetime
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd

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
    1. Python file job arguments:
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
# 4. Project setup
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

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print("Project root:", PROJECT_ROOT)


# ============================================================
# 5. Optional MixedLinear import
# ============================================================

MIXED_LINEAR_AVAILABLE = False
MixedLinear = None

candidate_roots = [
    PROJECT_ROOT,
    os.getcwd(),
    "/Workspace/Users/vadali.tejasviram@beebi-consulting.com/PE_work",
]

candidate_files = []

for root in candidate_roots:
    candidate_files.extend([
        os.path.join(root, "models", "mixed_linear_model.py"),
        os.path.join(root, "src", "priceelasticity", "scripts", "models", "mixed_linear_model.py"),
        os.path.join(root, "priceelasticity", "scripts", "models", "mixed_linear_model.py"),
    ])

print("Checking MixedLinear candidate files:")

for file_path in candidate_files:
    print(file_path, "exists:", os.path.exists(file_path))

for file_path in candidate_files:
    if os.path.exists(file_path):
        try:
            spec = importlib.util.spec_from_file_location(
                "mixed_linear_model",
                file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            MixedLinear = module.MixedLinear
            MIXED_LINEAR_AVAILABLE = True

            print("MixedLinear imported successfully from:", file_path)
            break

        except Exception as exc:
            print("Found file but failed to import:", file_path)
            print("Import error:", exc)

if not MIXED_LINEAR_AVAILABLE:
    print("MixedLinear import failed.")
    print("This notebook will use fixed-effect fallback for mixed_linear branches.")
    print("Checked these paths:")
    for file_path in candidate_files:
        print("  ", file_path)


# ============================================================
# 6. Result wrappers for fixed-effect causal model
# ============================================================

class ParameterContainer:
    def __init__(self, fe_params: Dict[str, float], random_effects_by_group: Dict = None):
        self.fe_params = fe_params
        self.random_effects_by_group = random_effects_by_group or {}


class SimpleCausalResults:
    def __init__(self, fe_params: Dict[str, float], model_type: str, random_effects_by_group: Dict = None):
        self.parameters = ParameterContainer(
            fe_params=fe_params,
            random_effects_by_group=random_effects_by_group or {},
        )
        self.model_type = model_type


# ============================================================
# 7. Config
# ============================================================

OUTPUT_SCHEMA = get_param("output_schema", "workspace.default")
TABLE_PREFIX = get_param("table_prefix", "")

MIN_HISTORY_POINTS_FOR_PANEL_TRAINING_DAYS = int(
    get_param("min_history_points_for_panel_training_days", "28")
)

MAX_PANDAS_ROWS = int(
    get_param("max_pandas_rows", "200000")
)


def table_name(base_name: str) -> str:
    if TABLE_PREFIX:
        return f"{OUTPUT_SCHEMA}.{TABLE_PREFIX}_{base_name}"
    return f"{OUTPUT_SCHEMA}.{base_name}"


def view_name(base_name: str) -> str:
    if TABLE_PREFIX:
        return f"{TABLE_PREFIX}_{base_name}"
    return base_name


def get_common_config() -> Dict:
    return {
        "input_table": table_name("pe_causal_features_dev"),

        # Daily grain identity:
        # Main row grain is date + article + store.
        # wm_yr_wk is kept only as helper information.
        "date_col": "date",
        "week_col": "wm_yr_wk",
        "article_col": "pe_article",
        "store_col": "pe_store_group",
        "group_col": "pe_article_store_group",

        "effect_col": "discount",
        "effect_power": 4,

        "valid_price_col": "valid_price_flag",

        "orthogonalize_when_possible": True,

        # Daily version:
        # Old weekly code used 4 observations.
        # For daily, 4 weeks roughly means 28 daily observations.
        "minimum_history_points_for_panel_training": MIN_HISTORY_POINTS_FOR_PANEL_TRAINING_DAYS,

        "candidate_control_cols": [
            "log_store_stock_quantity",
            "log_inventory_onhand_quantity",
            "snap",
            "calendar_month",
            "day_of_week",
            "day_of_month",
            "week_of_year",
            "is_weekend",
        ],

        "scenario_discounts": [
            0.00,
            0.05,
            0.075,
            0.10,
            0.15,
            0.20,
        ],

        "ridge_alpha": 1e-5,
    }


def get_branch_config(branch_name: str, model_family: str) -> Dict:
    """
    branch_name:
        sales
        probability

    model_family:
        causal_fixed_effect
        mixed_linear
    """

    config = get_common_config()

    if branch_name == "sales":
        target_col = "sold_qty_log"
        fallback_target_col = "outcome_quantity"
        valid_training_col = "valid_for_causal_training"
        target_scale = "log_quantity"

        # Sales can keep the flexible discount curve.
        effect_power = 4

    elif branch_name == "probability":
        # Probability branch trains on logit scale.
        target_col = "probability_logit_target"
        fallback_target_col = "probability_logit_target"
        valid_training_col = "valid_for_probability_training_with_price"
        target_scale = "logit_probability"

        # Stabilization:
        # Use only discount_power_1 for probability.
        effect_power = 1

    else:
        raise ValueError(f"Unsupported branch_name: {branch_name}")

    if model_family == "causal_fixed_effect":
        model_suffix = "causal"
        model_name = f"{branch_name}_causal_discount_model_daily"

    elif model_family == "mixed_linear":
        model_suffix = "mixedlinear"
        model_name = f"{branch_name}_mixedlinear_discount_model_daily"

    else:
        raise ValueError(f"Unsupported model_family: {model_family}")

    config.update({
        "branch_name": branch_name,
        "model_family": model_family,
        "model_name": model_name,
        "target_scale": target_scale,

        "target_col": target_col,
        "fallback_target_col": fallback_target_col,
        "valid_training_col": valid_training_col,
        "effect_power": effect_power,
        "require_valid_training_rows": True,

        "output_table": table_name(f"pe_{branch_name}_{model_suffix}_counterfactual"),
        "summary_table": table_name(f"pe_{branch_name}_{model_suffix}_summary"),

        "output_view": view_name(f"pe_{branch_name}_{model_suffix}_counterfactual_view"),
        "summary_view": view_name(f"pe_{branch_name}_{model_suffix}_summary_view"),
    })

    return config


# ============================================================
# 8. Helpers
# ============================================================

def choose_existing_column(
    pdf: pd.DataFrame,
    preferred_col: str,
    fallback_col: str,
) -> str:
    if preferred_col in pdf.columns:
        return preferred_col

    if fallback_col in pdf.columns:
        return fallback_col

    raise ValueError(
        f"Missing target column. Expected one of: {preferred_col}, {fallback_col}"
    )


def normalize_discount_values(
    pdf: pd.DataFrame,
    discount_col: str,
) -> pd.DataFrame:
    pdf[discount_col] = pd.to_numeric(pdf[discount_col], errors="coerce")
    pdf.loc[pdf[discount_col] < 0, discount_col] = 0.0
    pdf.loc[pdf[discount_col] > 0.80, discount_col] = 0.80
    return pdf


def create_discount_power_columns(
    pdf: pd.DataFrame,
    discount_col: str,
    max_power: int,
) -> Tuple[pd.DataFrame, List[str]]:
    power_cols = []

    for power in range(1, max_power + 1):
        col_name = f"{discount_col}_power_{power}"
        pdf[col_name] = np.power(pdf[discount_col].astype(float), power)
        power_cols.append(col_name)

    return pdf, power_cols


def find_available_columns(
    pdf: pd.DataFrame,
    candidate_cols: List[str],
) -> List[str]:
    return [
        col_name
        for col_name in candidate_cols
        if col_name in pdf.columns
    ]


def find_cce_columns(pdf: pd.DataFrame) -> List[str]:
    """
    Daily version:
    Prefer daily CCE columns.

    If daily CCE columns exist, use only daily CCE columns for
    orthogonalization.

    If daily CCE columns do not exist, fall back to whatever cce_* columns
    are available so the file remains backward-compatible.
    """

    daily_cce_cols = [
        col_name
        for col_name in pdf.columns
        if col_name.startswith("cce_") and "_day_" in col_name
    ]

    if daily_cce_cols:
        return daily_cce_cols

    return [
        col_name
        for col_name in pdf.columns
        if col_name.startswith("cce_")
    ]


def remove_constant_columns(
    pdf: pd.DataFrame,
    cols: List[str],
) -> List[str]:
    usable_cols = []

    for col_name in cols:
        if col_name not in pdf.columns:
            continue

        unique_count = pdf[col_name].nunique(dropna=True)

        if unique_count > 1:
            usable_cols.append(col_name)
        else:
            print("Removing constant column:", col_name)

    return usable_cols


def clean_numeric_training_data(
    pdf: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    clean_pdf = pdf.copy()

    numeric_cols = [target_col] + feature_cols

    for col_name in numeric_cols:
        clean_pdf[col_name] = pd.to_numeric(clean_pdf[col_name], errors="coerce")

    clean_pdf = clean_pdf.replace([np.inf, -np.inf], np.nan)
    clean_pdf = clean_pdf.dropna(subset=numeric_cols).reset_index(drop=True)

    return clean_pdf


def prepare_cce_columns(
    pdf: pd.DataFrame,
    cce_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    clean_pdf = pdf.copy()
    usable_cols = []

    for col_name in cce_cols:
        clean_pdf[col_name] = pd.to_numeric(clean_pdf[col_name], errors="coerce")
        clean_pdf[col_name] = clean_pdf[col_name].replace([np.inf, -np.inf], np.nan)

        non_null_count = clean_pdf[col_name].notna().sum()
        unique_count = clean_pdf[col_name].nunique(dropna=True)

        if non_null_count == 0 or unique_count <= 1:
            print("Removing unusable CCE column:", col_name)
            continue

        fill_value = clean_pdf[col_name].mean()
        clean_pdf[col_name] = clean_pdf[col_name].fillna(fill_value)

        usable_cols.append(col_name)

    return clean_pdf, usable_cols


def sigmoid_np(values):
    clipped_values = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped_values))


def validate_daily_feature_table(sdf, config: Dict):
    if "base_data_grain" in sdf.columns:
        grains = [
            row["base_data_grain"]
            for row in sdf.select("base_data_grain").distinct().collect()
        ]

        print("Feature table grain values:", grains)

        if "product_store_day" not in grains:
            raise ValueError(
                "This daily causal model expects product_store_day features. "
                f"Found base_data_grain values: {grains}. "
                "Run daily base data and daily feature engineering first."
            )

    duplicate_count = (
        sdf
        .groupBy(
            config["date_col"],
            config["article_col"],
            config["store_col"],
        )
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Feature table has duplicate product-store-date keys: {duplicate_count}. "
            "Expected one row per date + product + store."
        )


def run_pre_model_quality_gate(sdf, config: Dict):
    """
    Production safety gate before PE model training.

    It creates:
        <table_prefix>_pe_data_quality_gate_summary

    It fails only for hard data problems:
        - duplicate daily grain
        - no training rows
        - invalid discount values
    """

    print("==================================================")
    print("Running pre-model PE data quality gate")
    print("==================================================")

    date_col = config["date_col"]
    article_col = config["article_col"]
    store_col = config["store_col"]
    group_col = config["group_col"]
    effect_col = config["effect_col"]
    valid_col = config["valid_training_col"]

    total_rows = sdf.count()

    duplicate_count = (
        sdf
        .groupBy(date_col, article_col, store_col)
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    valid_training_rows = (
        sdf
        .filter(F.col(valid_col) == 1)
        .count()
    )

    invalid_price_rows = 0
    if "pe_unit_price" in sdf.columns:
        invalid_price_rows = (
            sdf
            .filter(
                (F.col("pe_unit_price").isNull()) |
                (F.col("pe_unit_price") <= 0)
            )
            .count()
        )

    invalid_discount_rows = (
        sdf
        .filter(
            (F.col(effect_col).isNull()) |
            (F.col(effect_col) < 0) |
            (F.col(effect_col) > 0.80)
        )
        .count()
    )

    group_quality = (
        sdf
        .filter(F.col(valid_col) == 1)
        .groupBy(group_col)
        .agg(
            F.count("*").alias("history_rows"),
            F.countDistinct(date_col).alias("days_count"),
            F.countDistinct(effect_col).alias("discount_variation_count")
        )
        .withColumn(
            "eligible_for_product_store_elasticity",
            F.when(
                (F.col("history_rows") >= F.lit(MIN_HISTORY_POINTS_FOR_PANEL_TRAINING_DAYS)) &
                (F.col("discount_variation_count") >= F.lit(2)),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
    )

    eligible_groups = (
        group_quality
        .filter(F.col("eligible_for_product_store_elasticity") == 1)
        .count()
    )

    total_groups = group_quality.count()

    summary_rows = [
        ("total_rows", str(total_rows)),
        ("valid_training_rows", str(valid_training_rows)),
        ("duplicate_product_store_date_rows", str(duplicate_count)),
        ("invalid_price_rows", str(invalid_price_rows)),
        ("invalid_discount_rows", str(invalid_discount_rows)),
        ("total_product_store_groups", str(total_groups)),
        ("eligible_product_store_elasticity_groups", str(eligible_groups)),
        ("min_history_required_days", str(MIN_HISTORY_POINTS_FOR_PANEL_TRAINING_DAYS)),
    ]

    summary_df = spark.createDataFrame(summary_rows, ["metric", "value"])
    summary_table = table_name("pe_data_quality_gate_summary")

    (
        summary_df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(summary_table)
    )

    print("Data quality gate summary saved:", summary_table)

    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate product-store-date rows found: {duplicate_count}. "
            "Fix base data before training PE models."
        )

    if valid_training_rows <= 0:
        raise ValueError(
            f"No valid rows for model training. Column checked: {valid_col}"
        )

    if invalid_discount_rows > 0:
        raise ValueError(
            f"Invalid discount rows found: {invalid_discount_rows}. "
            "Discount should be between 0 and 0.80."
        )

    if eligible_groups == 0:
        print("WARNING: No product-store group has enough discount variation.")
        print("Model will fall back to global elasticity effect.")

    print("Pre-model PE data quality gate completed.")


# ============================================================
# 9. Load and prepare data
# ============================================================

def load_causal_feature_data(config: Dict) -> pd.DataFrame:
    print("Loading causal feature table:", config["input_table"])

    sdf = spark.table(config["input_table"])

    validate_daily_feature_table(sdf, config)
    run_pre_model_quality_gate(sdf, config)

    valid_col = config["valid_training_col"]

    if valid_col not in sdf.columns:
        raise ValueError(
            f"Missing required valid training column: {valid_col}. "
            f"Available columns: {sdf.columns}"
        )

    # Filter in Spark before converting to Pandas.
    sdf = sdf.filter(F.col(valid_col) == 1)

    keep_cols = [
        config["date_col"],
        config["week_col"],
        config["article_col"],
        config["store_col"],
        config["group_col"],

        config["target_col"],
        config["fallback_target_col"],
        config["effect_col"],
        config["valid_training_col"],
        config["valid_price_col"],

        "base_data_grain",

        "pe_quantity",
        "pe_unit_price",
        "pe_actual_retail_price",
        "pe_average_zone_retail_price",

        "pe_category",
        "pe_product_type",
        "pe_product_division",
        "pe_country",

        "is_sold",
        "probability_target",
        "probability_target_smoothed",
        "probability_logit_target",

        "valid_for_causal_training",
        "valid_for_probability_training",
        "valid_for_probability_training_with_price",
        "valid_for_price_elasticity",
        "valid_for_mdo_input",

        "log_store_stock_quantity",
        "log_inventory_onhand_quantity",
        "snap",
        "calendar_month",
        "calendar_year",
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "is_weekend",

        "rows_per_product_store",
        "days_per_product_store",
        "weeks_per_product_store",
        "price_variation_count",
        "discount_variation_count",
    ]

    cce_cols = [
        col_name
        for col_name in sdf.columns
        if col_name.startswith("cce_")
    ]

    selected_cols = []
    for col_name in keep_cols + cce_cols:
        if col_name in sdf.columns and col_name not in selected_cols:
            selected_cols.append(col_name)

    sdf = sdf.select(*selected_cols)

    row_count = sdf.count()

    print("Rows after valid training filter:", row_count)
    print("Selected columns:", len(selected_cols))

    if row_count == 0:
        raise ValueError("Input causal feature table is empty after filtering.")

    if row_count > MAX_PANDAS_ROWS:
        fraction = MAX_PANDAS_ROWS / row_count

        print("Sampling causal training data to avoid Python memory crash.")
        print("Original rows:", row_count)
        print("Target max rows:", MAX_PANDAS_ROWS)
        print("Sample fraction:", fraction)

        sdf = (
            sdf
            .sample(withReplacement=False, fraction=fraction, seed=42)
            .limit(MAX_PANDAS_ROWS)
        )

    pdf = sdf.toPandas()

    print("Pandas rows loaded:", len(pdf))

    return pdf


def prepare_training_data(config: Dict) -> Dict:
    pdf = load_causal_feature_data(config)

    target_col = choose_existing_column(
        pdf=pdf,
        preferred_col=config["target_col"],
        fallback_col=config["fallback_target_col"],
    )

    config["active_target_col"] = target_col

    required_cols = [
        config["date_col"],
        config["article_col"],
        config["store_col"],
        config["group_col"],
        target_col,
        config["effect_col"],
        config["valid_training_col"],
    ]

    if config["branch_name"] == "probability":
        required_cols.extend([
            "probability_target",
            "probability_target_smoothed",
            "probability_logit_target",
        ])

    missing_cols = [
        col_name
        for col_name in required_cols
        if col_name not in pdf.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns for {config['branch_name']} {config['model_family']}: {missing_cols}"
        )

    pdf[config["date_col"]] = pd.to_datetime(pdf[config["date_col"]])

    valid_training_col = config["valid_training_col"]
    valid_training_rows = int(
        pd.to_numeric(pdf[valid_training_col], errors="coerce").fillna(0).sum()
    )

    print(f"Rows marked {valid_training_col}:", valid_training_rows)

    if config["require_valid_training_rows"]:
        if valid_training_rows <= 0:
            raise ValueError(
                f"No valid training rows found for {config['branch_name']} {config['model_family']}."
            )

        before_rows = len(pdf)
        pdf = pdf[pdf[valid_training_col] == 1].copy()
        valid_training_filter_used = True

        print(f"Applied {valid_training_col} filter:", len(pdf), "from", before_rows)

    else:
        valid_training_filter_used = False

        if config["valid_price_col"] in pdf.columns:
            before_rows = len(pdf)
            pdf = pdf[pdf[config["valid_price_col"]] == 1].copy()
            print("Rows after valid_price_flag filter:", len(pdf), "from", before_rows)

    pdf = normalize_discount_values(
        pdf=pdf,
        discount_col=config["effect_col"],
    )

    pdf = pdf.dropna(
        subset=[
            config["date_col"],
            config["group_col"],
            target_col,
            config["effect_col"],
        ]
    ).reset_index(drop=True)

    if pdf.empty:
        raise ValueError(
            f"No rows left after filtering for {config['branch_name']} {config['model_family']}."
        )

    pdf, discount_power_cols = create_discount_power_columns(
        pdf=pdf,
        discount_col=config["effect_col"],
        max_power=config["effect_power"],
    )

    control_cols = find_available_columns(
        pdf=pdf,
        candidate_cols=config["candidate_control_cols"],
    )

    feature_cols = discount_power_cols + control_cols
    feature_cols = remove_constant_columns(pdf, feature_cols)

    required_discount_feature = f"{config['effect_col']}_power_1"

    if required_discount_feature not in feature_cols:
        raise ValueError(
            f"{required_discount_feature} is missing or constant. "
            f"{config['branch_name']} {config['model_family']} cannot learn discount effect."
        )

    pdf = clean_numeric_training_data(
        pdf=pdf,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    if pdf.empty:
        raise ValueError(
            f"No rows left after numeric cleaning for {config['branch_name']} {config['model_family']}."
        )

    cce_cols = find_cce_columns(pdf)
    pdf, cce_cols = prepare_cce_columns(pdf, cce_cols)
    cce_cols = remove_constant_columns(pdf, cce_cols)

    group_counts = pdf.groupby(config["group_col"]).size()

    max_rows_per_group = int(group_counts.max())
    min_rows_per_group = int(group_counts.min())
    avg_rows_per_group = float(group_counts.mean())
    repeated_group_count = int((group_counts > 1).sum())

    can_use_panel_model = (
        max_rows_per_group >= config["minimum_history_points_for_panel_training"]
    )

    print("Final training rows:", len(pdf))
    print("Branch:", config["branch_name"])
    print("Model family:", config["model_family"])
    print("Target column:", target_col)
    print("Target scale:", config["target_scale"])
    print("Effect power:", config["effect_power"])
    print("Feature columns:", feature_cols)
    print("CCE columns:", cce_cols)
    print("Article-store groups:", pdf[config["group_col"]].nunique())
    print("Min rows per group:", min_rows_per_group)
    print("Max rows per group:", max_rows_per_group)
    print("Avg rows per group:", avg_rows_per_group)
    print("Groups with repeated rows:", repeated_group_count)
    print("Required daily history points:", config["minimum_history_points_for_panel_training"])
    print("Can use panel model:", can_use_panel_model)
    print("Date range:", pdf[config["date_col"]].min(), "to", pdf[config["date_col"]].max())

    return {
        "data": pdf,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "cce_cols": cce_cols,
        "min_rows_per_group": min_rows_per_group,
        "max_rows_per_group": max_rows_per_group,
        "avg_rows_per_group": avg_rows_per_group,
        "repeated_group_count": repeated_group_count,
        "can_use_panel_model": can_use_panel_model,
        "valid_training_filter_used": valid_training_filter_used,
    }


# ============================================================
# 10. CCE orthogonalization
# ============================================================

def can_run_orthogonalization(
    train_objects: Dict,
    config: Dict,
) -> bool:
    if not config["orthogonalize_when_possible"]:
        return False

    if not train_objects["can_use_panel_model"]:
        print("Skipping CCE orthogonalization because product-store groups do not have enough daily rows.")
        return False

    if not train_objects["cce_cols"]:
        print("Skipping CCE orthogonalization because no usable CCE columns exist.")
        return False

    return True


def orthogonalize_group(
    group_pdf: pd.DataFrame,
    target_cols: List[str],
    factor_cols: List[str],
) -> pd.DataFrame:
    available_factors = [
        col_name
        for col_name in factor_cols
        if col_name in group_pdf.columns
    ]

    available_targets = [
        col_name
        for col_name in target_cols
        if col_name in group_pdf.columns
    ]

    if not available_factors or not available_targets:
        return group_pdf

    for col_name in available_targets:
        group_pdf[col_name] = pd.to_numeric(group_pdf[col_name], errors="coerce").astype(float)

    x = group_pdf[available_factors].astype(float).to_numpy()
    y = group_pdf[available_targets].astype(float).to_numpy()

    projection = x @ np.linalg.pinv(x.T @ x) @ x.T
    orthogonal_values = y - projection @ y

    group_pdf.loc[:, available_targets] = orthogonal_values

    return group_pdf


def apply_cce_orthogonalization(
    train_objects: Dict,
    config: Dict,
) -> Tuple[pd.DataFrame, bool]:
    pdf = train_objects["data"].copy()

    if not can_run_orthogonalization(train_objects, config):
        return pdf, False

    print("Running daily CCE orthogonalization.")

    pdf["intercept"] = 1.0

    factor_cols = train_objects["cce_cols"] + ["intercept"]
    target_cols = [train_objects["target_col"]] + train_objects["feature_cols"]
    target_cols = list(dict.fromkeys(target_cols))

    model_pdf = (
        pdf
        .groupby(config["group_col"], group_keys=False)
        .apply(
            orthogonalize_group,
            target_cols=target_cols,
            factor_cols=factor_cols,
        )
        .reset_index(drop=True)
    )

    print("CCE orthogonalization completed.")

    return model_pdf, True


# ============================================================
# 11. Model training
# ============================================================

def train_fixed_effect_causal_model(
    model_pdf: pd.DataFrame,
    train_objects: Dict,
    config: Dict,
) -> SimpleCausalResults:
    print(f"Training fixed-effect causal model for {config['branch_name']}.")

    target_col = train_objects["target_col"]
    feature_cols = train_objects["feature_cols"]

    x = model_pdf[feature_cols].astype(float).to_numpy()
    y = model_pdf[target_col].astype(float).to_numpy()

    x_aug = np.column_stack([
        np.ones(len(x)),
        x,
    ])

    alpha = float(config["ridge_alpha"])

    penalty = np.eye(x_aug.shape[1]) * alpha
    penalty[0, 0] = 0.0

    beta = np.linalg.pinv(x_aug.T @ x_aug + penalty) @ x_aug.T @ y

    fe_params = {"intercept": float(beta[0])}

    for idx, col_name in enumerate(feature_cols, start=1):
        fe_params[col_name] = float(beta[idx])

    print("Fixed-effect causal model trained.")
    print("Fixed effect parameters:")

    for key, value in fe_params.items():
        print(key, "=", value)

    return SimpleCausalResults(
        fe_params=fe_params,
        model_type="causal_fixed_effect",
        random_effects_by_group={},
    )


def train_mixed_linear_model(
    model_pdf: pd.DataFrame,
    train_objects: Dict,
    config: Dict,
):
    """
    Try MixedLinear first.

    If MixedLinear is not available or fails, use fixed-effect fallback.
    This prevents the notebook from failing and still creates the required
    mixedlinear output tables used by predictive_model_dev.py.

    Daily version:
    date_col is now exact date, not weekly date.
    """

    if not MIXED_LINEAR_AVAILABLE:
        print("WARNING: MixedLinear is not available.")
        print("Using fixed-effect fallback for mixed_linear branch.")
        print("Branch:", config["branch_name"])

        fallback_results = train_fixed_effect_causal_model(
            model_pdf=model_pdf,
            train_objects=train_objects,
            config=config,
        )

        fallback_results.model_type = "mixed_linear_fallback_fixed_effect"

        return fallback_results

    if not train_objects["can_use_panel_model"]:
        print("WARNING: Panel data is not sufficient for MixedLinear.")
        print("Using fixed-effect fallback for mixed_linear branch.")
        print("Branch:", config["branch_name"])

        fallback_results = train_fixed_effect_causal_model(
            model_pdf=model_pdf,
            train_objects=train_objects,
            config=config,
        )

        fallback_results.model_type = "mixed_linear_fallback_fixed_effect"

        return fallback_results

    try:
        print(f"Training MixedLinear model for {config['branch_name']}.")

        random_effect_df = model_pdf[[config["effect_col"]]]

        model = MixedLinear(
            exog=model_pdf[train_objects["feature_cols"]],
            group_col=model_pdf[config["group_col"]],
            endog=model_pdf[train_objects["target_col"]],
            random_effects=random_effect_df,
            date_col=model_pdf[config["date_col"]],
        )

        results = model.train(est_re_cov=False)
        results.model_type = "mixed_linear"

        print("MixedLinear model trained.")
        print("Fixed effect parameters:")

        for key, value in results.parameters.fe_params.items():
            print(key, "=", value)

        return results

    except Exception as exc:
        print("WARNING: MixedLinear training failed.")
        print("Error:", exc)
        print("Using fixed-effect fallback for mixed_linear branch.")
        print("Branch:", config["branch_name"])

        fallback_results = train_fixed_effect_causal_model(
            model_pdf=model_pdf,
            train_objects=train_objects,
            config=config,
        )

        fallback_results.model_type = "mixed_linear_fallback_fixed_effect"

        return fallback_results


def train_model_by_family(
    model_pdf: pd.DataFrame,
    train_objects: Dict,
    config: Dict,
):
    if config["model_family"] == "causal_fixed_effect":
        return train_fixed_effect_causal_model(
            model_pdf=model_pdf,
            train_objects=train_objects,
            config=config,
        )

    if config["model_family"] == "mixed_linear":
        return train_mixed_linear_model(
            model_pdf=model_pdf,
            train_objects=train_objects,
            config=config,
        )

    raise ValueError(f"Unsupported model_family: {config['model_family']}")


# ============================================================
# 12. Counterfactual helpers
# ============================================================

def polynomial_value(
    coefs: List[float],
    x_value: float,
) -> float:
    total = 0.0

    for idx, coef in enumerate(coefs, start=1):
        total += coef * (x_value ** idx)

    return float(total)


def get_discount_coefs(
    results,
    config: Dict,
) -> List[float]:
    coefs = []

    for power in range(1, config["effect_power"] + 1):
        param_name = f"{config['effect_col']}_power_{power}"
        value = float(results.parameters.fe_params.get(param_name, 0.0))
        coefs.append(value)

    return coefs


def get_random_discount_effect_for_group(
    results,
    group_value,
) -> Tuple[float, int, str]:
    """
    Reads product-store random discount effect from MixedLinear results.

    Different MixedLinear implementations can store random effects differently.
    This function is defensive:
        - dict by group
        - nested dict
        - scalar
        - missing random effects
    """

    random_effects_by_group = getattr(
        getattr(results, "parameters", None),
        "random_effects_by_group",
        {}
    )

    if not random_effects_by_group:
        return 0.0, 0, "no_random_effects_found"

    keys_to_try = [
        group_value,
        str(group_value),
    ]

    selected_random_effect = None

    for key in keys_to_try:
        if key in random_effects_by_group:
            selected_random_effect = random_effects_by_group.get(key)
            break

    if selected_random_effect is None:
        return 0.0, 0, "group_random_effect_missing"

    if isinstance(selected_random_effect, dict):
        for possible_key in [
            "random_discount",
            "discount",
            "discount_power_1",
            "random_effect_discount",
            "x_re_discount",
        ]:
            if possible_key in selected_random_effect:
                value = selected_random_effect.get(possible_key)
                try:
                    value = float(value)
                    return value, 1 if abs(value) > 0 else 0, "product_store_random_effect"
                except Exception:
                    continue

        numeric_values = []
        for value in selected_random_effect.values():
            try:
                numeric_values.append(float(value))
            except Exception:
                pass

        if numeric_values:
            value = float(numeric_values[0])
            return value, 1 if abs(value) > 0 else 0, "product_store_random_effect_first_numeric"

        return 0.0, 0, "random_effect_dict_without_numeric_value"

    try:
        value = float(selected_random_effect)
        return value, 1 if abs(value) > 0 else 0, "product_store_random_effect_scalar"
    except Exception:
        return 0.0, 0, "random_effect_unreadable"


def calculate_direct_elasticity_for_group(group_pdf: pd.DataFrame) -> Dict:
    """
    Estimate direct product-store elasticity using a log-log relationship.

    Formula:
        log(quantity) = intercept + elasticity * log(price / base_price)

    For markdown pricing, a good elasticity is normally negative.
    """

    work_pdf = group_pdf.copy()

    required_cols = [
        "pe_quantity",
        "pe_unit_price",
        "pe_average_zone_retail_price",
    ]

    for col_name in required_cols:
        if col_name not in work_pdf.columns:
            return {
                "estimated_elasticity": None,
                "elasticity_source": "missing_required_columns",
                "elasticity_status": "fallback_needed",
            }

    work_pdf["pe_quantity"] = pd.to_numeric(
        work_pdf["pe_quantity"],
        errors="coerce"
    )
    work_pdf["pe_unit_price"] = pd.to_numeric(
        work_pdf["pe_unit_price"],
        errors="coerce"
    )
    work_pdf["pe_average_zone_retail_price"] = pd.to_numeric(
        work_pdf["pe_average_zone_retail_price"],
        errors="coerce"
    )

    work_pdf = work_pdf[
        (work_pdf["pe_quantity"] > 0) &
        (work_pdf["pe_unit_price"] > 0) &
        (work_pdf["pe_average_zone_retail_price"] > 0)
    ].copy()

    if len(work_pdf) < 28:
        return {
            "estimated_elasticity": None,
            "elasticity_source": "insufficient_history",
            "elasticity_status": "fallback_needed",
        }

    work_pdf["log_quantity"] = np.log(work_pdf["pe_quantity"])

    work_pdf["log_price_ratio"] = np.log(
        work_pdf["pe_unit_price"] /
        work_pdf["pe_average_zone_retail_price"]
    )

    work_pdf = work_pdf.replace([np.inf, -np.inf], np.nan)
    work_pdf = work_pdf.dropna(
        subset=[
            "log_quantity",
            "log_price_ratio",
        ]
    )

    if len(work_pdf) < 28:
        return {
            "estimated_elasticity": None,
            "elasticity_source": "insufficient_clean_history",
            "elasticity_status": "fallback_needed",
        }

    if work_pdf["log_price_ratio"].nunique() < 2:
        return {
            "estimated_elasticity": None,
            "elasticity_source": "insufficient_price_variation",
            "elasticity_status": "fallback_needed",
        }

    x = work_pdf["log_price_ratio"].astype(float).to_numpy()
    y = work_pdf["log_quantity"].astype(float).to_numpy()

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    denominator = float(np.sum((x - x_mean) ** 2))

    if denominator <= 0:
        return {
            "estimated_elasticity": None,
            "elasticity_source": "zero_price_variance",
            "elasticity_status": "fallback_needed",
        }

    numerator = float(np.sum((x - x_mean) * (y - y_mean)))
    elasticity = numerator / denominator

    if elasticity >= 0:
        return {
            "estimated_elasticity": None,
            "elasticity_source": "positive_or_invalid_elasticity",
            "elasticity_status": "fallback_needed",
        }

    elasticity = max(-5.0, min(-0.05, float(elasticity)))

    return {
        "estimated_elasticity": elasticity,
        "elasticity_source": "product_store_log_log",
        "elasticity_status": "direct_estimate",
    }


def create_sales_discount_coefficients_table(
    training_pdf: pd.DataFrame,
    results,
    config: Dict,
):
    """
    Creates product-store level sales discount coefficients.

    Main PE correction:
    Instead of using only one global discount coefficient,
    this creates one coefficient row per pe_article_store_group.

    Output:
        <table_prefix>_pe_sales_discount_coefficients
    """

    if config["branch_name"] != "sales":
        return None

    print("==================================================")
    print("Creating product-store sales discount coefficient table")
    print("==================================================")

    group_col = config["group_col"]
    article_col = config["article_col"]
    store_col = config["store_col"]
    effect_col = config["effect_col"]

    global_coefs = get_discount_coefs(results, config)

    while len(global_coefs) < 4:
        global_coefs.append(0.0)

    base_cols = [
        group_col,
        article_col,
        store_col,
    ]

    optional_cols = [
        "pe_category",
        "pe_product_type",
        "pe_product_division",
        "pe_country",
    ]

    for col_name in optional_cols:
        if col_name in training_pdf.columns:
            base_cols.append(col_name)

    base_info = (
        training_pdf[base_cols]
        .drop_duplicates(subset=[group_col])
        .copy()
    )

    group_stats = (
        training_pdf
        .groupby(group_col)
        .agg(
            history_rows=(group_col, "count"),
            discount_variation_count=(effect_col, "nunique")
        )
        .reset_index()
    )

    coeff_pdf = base_info.merge(group_stats, on=group_col, how="left")

    rows = []

    for _, row in coeff_pdf.iterrows():
        group_value = row[group_col]

        group_training_pdf = training_pdf[
            training_pdf[group_col] == group_value
        ].copy()

        direct_elasticity_result = calculate_direct_elasticity_for_group(
            group_training_pdf
        )

        estimated_elasticity = direct_elasticity_result["estimated_elasticity"]
        elasticity_source = direct_elasticity_result["elasticity_source"]
        elasticity_status = direct_elasticity_result["elasticity_status"]

        random_discount, random_effect_applied, coefficient_source = (
            get_random_discount_effect_for_group(
                results=results,
                group_value=group_value,
            )
        )

        coefficient_1 = float(global_coefs[0]) + float(random_discount)
        coefficient_2 = float(global_coefs[1])
        coefficient_3 = float(global_coefs[2])
        coefficient_4 = float(global_coefs[3])

        if estimated_elasticity is None:
            # Safe fallback when direct elasticity is not reliable.
            # This keeps the model usable for generic datasets,
            # but direct log-log elasticity remains the preferred source.
            approx_elasticity = -1.0 * abs(float(coefficient_1) / 0.20)

            estimated_elasticity = max(-5.0, min(-0.05, approx_elasticity))
            elasticity_source = "fallback_from_discount_coefficient"
            elasticity_status = "fallback_estimate"

        coefficient_level = (
            "product_store"
            if random_effect_applied == 1
            else "global_fallback"
        )

        model_status = (
            "using_product_store_random_effect"
            if random_effect_applied == 1
            else "using_global_fixed_effect_fallback"
        )

        rows.append({
            "pe_article": row[article_col],
            "pe_store_group": row[store_col],
            "pe_article_store_group": row[group_col],

            "pe_category": row["pe_category"] if "pe_category" in row else None,
            "pe_product_type": row["pe_product_type"] if "pe_product_type" in row else None,
            "pe_product_division": row["pe_product_division"] if "pe_product_division" in row else None,
            "pe_country": row["pe_country"] if "pe_country" in row else None,

            "coefficient_level": coefficient_level,
            "coefficient_source": coefficient_source,
            "model_status": model_status,

            "discount_effect_coefficient_1": coefficient_1,
            "discount_effect_coefficient_2": coefficient_2,
            "discount_effect_coefficient_3": coefficient_3,
            "discount_effect_coefficient_4": coefficient_4,

            "estimated_elasticity": float(estimated_elasticity),
            "elasticity_source": elasticity_source,
            "elasticity_status": elasticity_status,

            "global_discount_effect_coefficient_1": float(global_coefs[0]),
            "random_discount_effect_coefficient_1": float(random_discount),

            "history_rows": int(row["history_rows"]) if pd.notna(row["history_rows"]) else 0,
            "discount_variation_count": int(row["discount_variation_count"]) if pd.notna(row["discount_variation_count"]) else 0,
            "random_effect_applied": int(random_effect_applied),

            "model_type": getattr(results, "model_type", "unknown"),
            "model_family": config["model_family"],
            "branch_name": config["branch_name"],
            "model_created_at": datetime.datetime.now(datetime.timezone.utc),
        })

    output_pdf = pd.DataFrame(rows)

    output_table = table_name("pe_sales_discount_coefficients")
    output_sdf = spark.createDataFrame(output_pdf)

    (
        output_sdf.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(output_table)
    )

    print("Sales discount coefficient table saved:", output_table)

    display(
        spark.sql(f"""
            SELECT
                coefficient_level,
                model_status,
                COUNT(*) AS rows,
                AVG(discount_effect_coefficient_1) AS avg_coef_1,
                MIN(discount_effect_coefficient_1) AS min_coef_1,
                MAX(discount_effect_coefficient_1) AS max_coef_1,
                STDDEV(discount_effect_coefficient_1) AS stddev_coef_1,
                AVG(estimated_elasticity) AS avg_estimated_elasticity,
                MIN(estimated_elasticity) AS min_estimated_elasticity,
                MAX(estimated_elasticity) AS max_estimated_elasticity,
                STDDEV(estimated_elasticity) AS stddev_estimated_elasticity
            FROM {output_table}
            GROUP BY coefficient_level, model_status
        """)
    )

    return output_table


# ============================================================
# 13. Sales counterfactual
# ============================================================

def create_sales_counterfactual_table(
    training_pdf: pd.DataFrame,
    results,
    config: Dict,
) -> pd.DataFrame:
    print("Creating daily sales counterfactual table.")

    target_col = config["active_target_col"]
    effect_col = config["effect_col"]

    coefs = get_discount_coefs(results, config)

    base_cols = [
        config["date_col"],
        config["week_col"],
        config["article_col"],
        config["store_col"],
        config["group_col"],
        target_col,
        effect_col,
    ]

    optional_cols = [
        "base_data_grain",
        "pe_quantity",
        "pe_unit_price",
        "pe_actual_retail_price",
        "pe_average_zone_retail_price",
        "pe_category",
        "pe_product_type",
        "pe_product_division",
        "pe_country",
        "valid_for_causal_training",
        "valid_for_price_elasticity",
        "valid_for_mdo_input",
    ]

    for col_name in optional_cols:
        if col_name in training_pdf.columns and col_name not in base_cols:
            base_cols.append(col_name)

    base_df = training_pdf[base_cols].copy()

    base_df["actual_quantity"] = np.exp(base_df[target_col])

    base_df["current_discount_effect"] = base_df[effect_col].apply(
        lambda x: polynomial_value(coefs, x)
    )

    base_df["base_log_quantity_without_discount"] = (
        base_df[target_col] - base_df["current_discount_effect"]
    )

    scenario_df = pd.DataFrame({
        "scenario_discount": config["scenario_discounts"]
    })

    base_df["_join_key"] = 1
    scenario_df["_join_key"] = 1

    output_df = (
        base_df
        .merge(scenario_df, on="_join_key", how="inner")
        .drop(columns=["_join_key"])
    )

    output_df["scenario_discount_effect"] = output_df["scenario_discount"].apply(
        lambda x: polynomial_value(coefs, x)
    )

    output_df["counterfactual_log_quantity"] = (
        output_df["base_log_quantity_without_discount"]
        + output_df["scenario_discount_effect"]
    )

    output_df["counterfactual_quantity"] = np.exp(
        output_df["counterfactual_log_quantity"]
    )

    output_df["quantity_change"] = (
        output_df["counterfactual_quantity"]
        - output_df["actual_quantity"]
    )

    output_df["quantity_change_pct"] = np.where(
        output_df["actual_quantity"] != 0,
        output_df["quantity_change"] / output_df["actual_quantity"],
        np.nan,
    )

    output_df["discount_effect_coefficient_1"] = coefs[0] if len(coefs) > 0 else 0.0
    output_df["discount_effect_coefficient_2"] = coefs[1] if len(coefs) > 1 else 0.0
    output_df["discount_effect_coefficient_3"] = coefs[2] if len(coefs) > 2 else 0.0
    output_df["discount_effect_coefficient_4"] = coefs[3] if len(coefs) > 3 else 0.0

    output_df["model_type"] = getattr(results, "model_type", "unknown")
    output_df["model_family"] = config["model_family"]
    output_df["branch_name"] = config["branch_name"]
    output_df["target_scale"] = config["target_scale"]
    output_df["model_created_at"] = datetime.datetime.now(datetime.timezone.utc)

    final_cols = [
        "branch_name",
        "model_family",
        "model_type",
        "target_scale",
        config["date_col"],
        config["week_col"],
        config["article_col"],
        config["store_col"],
        config["group_col"],
        effect_col,
        "scenario_discount",
        "actual_quantity",
        "counterfactual_quantity",
        "quantity_change",
        "quantity_change_pct",
        "current_discount_effect",
        "scenario_discount_effect",
        "discount_effect_coefficient_1",
        "discount_effect_coefficient_2",
        "discount_effect_coefficient_3",
        "discount_effect_coefficient_4",
        "model_created_at",
    ]

    for col_name in optional_cols:
        if col_name in output_df.columns and col_name not in final_cols:
            final_cols.append(col_name)

    final_df = output_df[final_cols].copy()

    print("Sales counterfactual rows:", len(final_df))

    return final_df


# ============================================================
# 14. Probability counterfactual on logit scale
# ============================================================

def create_probability_counterfactual_table(
    training_pdf: pd.DataFrame,
    results,
    config: Dict,
) -> pd.DataFrame:
    print("Creating daily probability counterfactual table on logit scale.")

    target_col = config["active_target_col"]
    effect_col = config["effect_col"]

    coefs = get_discount_coefs(results, config)

    base_cols = [
        config["date_col"],
        config["week_col"],
        config["article_col"],
        config["store_col"],
        config["group_col"],
        target_col,
        effect_col,
    ]

    optional_cols = [
        "base_data_grain",
        "probability_target",
        "probability_target_smoothed",
        "pe_quantity",
        "pe_unit_price",
        "pe_actual_retail_price",
        "pe_average_zone_retail_price",
        "pe_category",
        "pe_product_type",
        "pe_product_division",
        "pe_country",
        "is_sold",
        "valid_for_probability_training",
        "valid_for_probability_training_with_price",
        "valid_for_price_elasticity",
        "valid_for_mdo_input",
    ]

    for col_name in optional_cols:
        if col_name in training_pdf.columns and col_name not in base_cols:
            base_cols.append(col_name)

    base_df = training_pdf[base_cols].copy()

    base_df["actual_probability_logit"] = pd.to_numeric(
        base_df[target_col],
        errors="coerce"
    )

    base_df["actual_probability"] = sigmoid_np(
        base_df["actual_probability_logit"]
    )

    if "probability_target" in base_df.columns:
        base_df["observed_probability_target"] = pd.to_numeric(
            base_df["probability_target"],
            errors="coerce"
        )
    else:
        base_df["observed_probability_target"] = base_df["actual_probability"]

    base_df["current_discount_effect"] = base_df[effect_col].apply(
        lambda x: polynomial_value(coefs, x)
    )

    base_df["base_logit_probability_without_discount"] = (
        base_df["actual_probability_logit"] - base_df["current_discount_effect"]
    )

    scenario_df = pd.DataFrame({
        "scenario_discount": config["scenario_discounts"]
    })

    base_df["_join_key"] = 1
    scenario_df["_join_key"] = 1

    output_df = (
        base_df
        .merge(scenario_df, on="_join_key", how="inner")
        .drop(columns=["_join_key"])
    )

    output_df["scenario_discount_effect"] = output_df["scenario_discount"].apply(
        lambda x: polynomial_value(coefs, x)
    )

    output_df["counterfactual_probability_logit"] = (
        output_df["base_logit_probability_without_discount"]
        + output_df["scenario_discount_effect"]
    )

    output_df["counterfactual_probability"] = sigmoid_np(
        output_df["counterfactual_probability_logit"]
    )

    output_df["raw_counterfactual_probability"] = output_df["counterfactual_probability"]

    output_df["probability_change"] = (
        output_df["counterfactual_probability"]
        - output_df["actual_probability"]
    )

    output_df["probability_change_pct"] = np.where(
        output_df["actual_probability"] != 0,
        output_df["probability_change"] / output_df["actual_probability"],
        np.nan,
    )

    output_df["probability_change_pp"] = output_df["probability_change"] * 100.0

    output_df["discount_effect_coefficient_1"] = coefs[0] if len(coefs) > 0 else 0.0
    output_df["discount_effect_coefficient_2"] = coefs[1] if len(coefs) > 1 else 0.0
    output_df["discount_effect_coefficient_3"] = coefs[2] if len(coefs) > 2 else 0.0
    output_df["discount_effect_coefficient_4"] = coefs[3] if len(coefs) > 3 else 0.0

    output_df["model_type"] = getattr(results, "model_type", "unknown")
    output_df["model_family"] = config["model_family"]
    output_df["branch_name"] = config["branch_name"]
    output_df["target_scale"] = config["target_scale"]
    output_df["model_created_at"] = datetime.datetime.now(datetime.timezone.utc)

    final_cols = [
        "branch_name",
        "model_family",
        "model_type",
        "target_scale",
        config["date_col"],
        config["week_col"],
        config["article_col"],
        config["store_col"],
        config["group_col"],
        effect_col,
        "scenario_discount",
        "observed_probability_target",
        "actual_probability",
        "actual_probability_logit",
        "base_logit_probability_without_discount",
        "counterfactual_probability",
        "counterfactual_probability_logit",
        "raw_counterfactual_probability",
        "probability_change",
        "probability_change_pct",
        "probability_change_pp",
        "current_discount_effect",
        "scenario_discount_effect",
        "discount_effect_coefficient_1",
        "discount_effect_coefficient_2",
        "discount_effect_coefficient_3",
        "discount_effect_coefficient_4",
        "model_created_at",
    ]

    for col_name in optional_cols:
        if col_name in output_df.columns and col_name not in final_cols:
            final_cols.append(col_name)

    final_df = output_df[final_cols].copy()

    print("Probability counterfactual rows:", len(final_df))

    return final_df


# ============================================================
# 15. Summary
# ============================================================

def create_summary_table(
    train_objects: Dict,
    model_pdf: pd.DataFrame,
    results,
    orthogonalized: bool,
    config: Dict,
) -> pd.DataFrame:
    coefs = get_discount_coefs(results, config)
    model_type = getattr(results, "model_type", "unknown")

    rows = [
        ("branch_name", config["branch_name"]),
        ("model_family", config["model_family"]),
        ("model_name", config["model_name"]),
        ("model_type", model_type),
        ("target_scale", config["target_scale"]),
        ("input_table", config["input_table"]),
        ("target_column", train_objects["target_col"]),
        ("effect_column", config["effect_col"]),
        ("effect_power", str(config["effect_power"])),
        ("training_filter_column", config["valid_training_col"]),
        ("training_rows", str(len(train_objects["data"]))),
        ("model_rows", str(len(model_pdf))),
        ("article_store_groups", str(train_objects["data"][config["group_col"]].nunique())),
        ("min_rows_per_group", str(train_objects["min_rows_per_group"])),
        ("max_rows_per_group", str(train_objects["max_rows_per_group"])),
        ("avg_rows_per_group", str(train_objects["avg_rows_per_group"])),
        ("repeated_group_count", str(train_objects["repeated_group_count"])),
        ("minimum_history_points_for_panel_training_days", str(config["minimum_history_points_for_panel_training"])),
        ("can_use_panel_model", str(train_objects["can_use_panel_model"])),
        ("min_date", str(train_objects["data"][config["date_col"]].min())),
        ("max_date", str(train_objects["data"][config["date_col"]].max())),
        ("feature_columns", str(train_objects["feature_cols"])),
        ("cce_columns", str(train_objects["cce_cols"])),
        ("orthogonalized", str(orthogonalized)),
        ("valid_training_filter_used", str(train_objects["valid_training_filter_used"])),
        ("discount_coefficients", str(coefs)),
        ("output_table", config["output_table"]),
    ]

    if train_objects["can_use_panel_model"]:
        rows.append(
            (
                "note",
                "Panel model path available: repeated product-store-day rows exist for product-store groups."
            )
        )
    else:
        rows.append(
            (
                "note",
                "Panel model path unavailable: not enough repeated product-store-day rows."
            )
        )

    if config["branch_name"] == "probability":
        rows.append(
            (
                "probability_model_note",
                "Probability branch is trained on probability_logit_target and converted back using sigmoid."
            )
        )

    if model_type == "mixed_linear_fallback_fixed_effect":
        rows.append(
            (
                "mixed_linear_fallback_note",
                "MixedLinear was unavailable or failed, so this mixed_linear branch used fixed-effect fallback."
            )
        )

    return pd.DataFrame(rows, columns=["metric", "value"])


# ============================================================
# 16. Runner
# ============================================================

def create_counterfactual_table(
    training_pdf: pd.DataFrame,
    results,
    config: Dict,
) -> pd.DataFrame:
    if config["branch_name"] == "sales":
        return create_sales_counterfactual_table(
            training_pdf=training_pdf,
            results=results,
            config=config,
        )

    if config["branch_name"] == "probability":
        return create_probability_counterfactual_table(
            training_pdf=training_pdf,
            results=results,
            config=config,
        )

    raise ValueError(f"Unsupported branch_name: {config['branch_name']}")


def run_model_branch(config: Dict):
    print("==================================================")
    print("Running DAILY causal model branch")
    print("Branch:", config["branch_name"])
    print("Model family:", config["model_family"])
    print("Input:", config["input_table"])
    print("Output:", config["output_table"])
    print("==================================================")

    train_objects = prepare_training_data(config)

    model_pdf, orthogonalized = apply_cce_orthogonalization(
        train_objects=train_objects,
        config=config,
    )

    active_run = mlflow.active_run()

    if active_run is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=config["model_name"]):
        mlflow.log_param("branch_name", config["branch_name"])
        mlflow.log_param("model_family", config["model_family"])
        mlflow.log_param("target_scale", config["target_scale"])
        mlflow.log_param("input_table", config["input_table"])
        mlflow.log_param("target_col", train_objects["target_col"])
        mlflow.log_param("effect_col", config["effect_col"])
        mlflow.log_param("effect_power", config["effect_power"])
        mlflow.log_param("training_rows", len(train_objects["data"]))
        mlflow.log_param("orthogonalized", orthogonalized)
        mlflow.log_param("can_use_panel_model", train_objects["can_use_panel_model"])
        mlflow.log_param("daily_grain", "product_store_day")
        mlflow.log_param(
            "minimum_history_points_for_panel_training_days",
            config["minimum_history_points_for_panel_training"]
        )

        results = train_model_by_family(
            model_pdf=model_pdf,
            train_objects=train_objects,
            config=config,
        )

        mlflow.log_param("actual_model_type", getattr(results, "model_type", "unknown"))

        counterfactual_pdf = create_counterfactual_table(
            training_pdf=train_objects["data"],
            results=results,
            config=config,
        )

        summary_pdf = create_summary_table(
            train_objects=train_objects,
            model_pdf=model_pdf,
            results=results,
            orthogonalized=orthogonalized,
            config=config,
        )

        counterfactual_sdf = spark.createDataFrame(counterfactual_pdf)
        summary_sdf = spark.createDataFrame(summary_pdf)

        (
            counterfactual_sdf.write
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(config["output_table"])
        )

        (
            summary_sdf.write
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(config["summary_table"])
        )

        if config["branch_name"] == "sales":
            create_sales_discount_coefficients_table(
                training_pdf=train_objects["data"],
                results=results,
                config=config,
            )

        counterfactual_sdf.createOrReplaceTempView(config["output_view"])
        summary_sdf.createOrReplaceTempView(config["summary_view"])

        mlflow.log_metric("counterfactual_rows", counterfactual_sdf.count())

    print("==================================================")
    print("Model branch completed")
    print("Branch:", config["branch_name"])
    print("Model family:", config["model_family"])
    print("Output table:", config["output_table"])
    print("Summary table:", config["summary_table"])
    print("==================================================")

    print("Summary:")
    display(
        spark.sql(f"""
            SELECT *
            FROM {config["summary_table"]}
        """)
    )

    print("Counterfactual sample:")
    display(
        spark.sql(f"""
            SELECT *
            FROM {config["output_table"]}
            ORDER BY pe_store_group, pe_article, date, scenario_discount
            LIMIT 100
        """)
    )

    print("Validation: daily counterfactual grain check")
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
                FROM {config["output_table"]}
                GROUP BY
                    date,
                    pe_article,
                    pe_store_group,
                    scenario_discount
                HAVING COUNT(*) > 1
            )
        """)
    )

    return results, counterfactual_sdf, summary_sdf


# ============================================================
# 17. Full runner
# ============================================================

def run_all_model_branches():
    print("==================================================")
    print("Running all required DAILY causal model branches")
    print("==================================================")
    print("Output schema:", OUTPUT_SCHEMA)
    print("Table prefix:", TABLE_PREFIX)
    print("Minimum daily history:", MIN_HISTORY_POINTS_FOR_PANEL_TRAINING_DAYS)

    configs = [
        get_branch_config("sales", "mixed_linear"),
        get_branch_config("probability", "mixed_linear"),
    ]

    outputs = {}

    for config in configs:
        key = f"{config['branch_name']}_{config['model_family']}"

        results, counterfactual_sdf, summary_sdf = run_model_branch(config)

        outputs[key] = {
            "results": results,
            "counterfactual_sdf": counterfactual_sdf,
            "summary_sdf": summary_sdf,
            "output_table": config["output_table"],
            "summary_table": config["summary_table"],
        }

    print("==================================================")
    print("All required DAILY causal model branches completed")
    print("==================================================")

    for key, value in outputs.items():
        print(key)
        print("  output:", value["output_table"])
        print("  summary:", value["summary_table"])

    return outputs


# ============================================================
# 18. Execute
# ============================================================

if __name__ == "__main__":
    model_outputs = run_all_model_branches()