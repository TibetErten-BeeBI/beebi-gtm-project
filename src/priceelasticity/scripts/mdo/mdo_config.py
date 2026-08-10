import sys
from typing import List


# ============================================================
# 1. Parameter reader
# ============================================================

def get_param(param_name: str, default_value: str = "") -> str:
    """
    Reads parameter from:
    1. Databricks Python file job arguments: --param_name value
    2. Databricks Python file job arguments: param_name value
    3. Databricks widgets
    4. Default value
    """
    arg_key_dash = f"--{param_name}"
    arg_key_plain = param_name

    if arg_key_dash in sys.argv:
        arg_index = sys.argv.index(arg_key_dash)
        if arg_index + 1 < len(sys.argv):
            return sys.argv[arg_index + 1].strip()

    if arg_key_plain in sys.argv:
        arg_index = sys.argv.index(arg_key_plain)
        if arg_index + 1 < len(sys.argv):
            return sys.argv[arg_index + 1].strip()

    for arg in sys.argv:
        arg = str(arg).strip()

        if arg.startswith(arg_key_dash + "="):
            return arg.split("=", 1)[1].strip()

        if arg.startswith(arg_key_plain + "="):
            return arg.split("=", 1)[1].strip()

    try:
        return dbutils.widgets.get(param_name).strip()
    except Exception:
        return default_value

def get_float_param(param_name: str, default_value: str) -> float:
    value = get_param(param_name, default_value)

    try:
        return float(value)
    except Exception:
        raise ValueError(f"Invalid float value for parameter {param_name}: {value}")


def parse_discount_list(discount_text: str) -> List[float]:
    if not discount_text:
        return []

    discounts = []

    for value in discount_text.split(","):
        value = value.strip()

        if value == "":
            continue

        discounts.append(float(value))

    return discounts


# ============================================================
# 2. Schema and table naming config
# ============================================================

OUTPUT_SCHEMA = get_param("output_schema", "workspace.default")

TABLE_PREFIX = get_param("table_prefix", "")


def table_name(base_name: str) -> str:
    """
    Creates generic table name.
    """
    if TABLE_PREFIX:
        return f"{OUTPUT_SCHEMA}.{TABLE_PREFIX}_{base_name}"

    return f"{OUTPUT_SCHEMA}.{base_name}"


# ============================================================
# 3. Input tables
# ============================================================

PE_ENGINE_OUTPUT_TABLE = table_name("pe_price_elasticity_engine_output_dev")

PREDICTIVE_SCENARIO_OUTPUT_TABLE = table_name("pe_predictive_scenario_output")

BASE_DATA_QUALITY_SUMMARY_TABLE = table_name("base_data_quality_summary")


# ============================================================
# 4. Output tables
# ============================================================

MDO_SCENARIO_RULE_OUTPUT_TABLE = table_name("mdo_phase1_scenario_rule_output")

MDO_FINAL_RECOMMENDATION_TABLE = table_name("mdo_phase1_final_recommendation")

MDO_QUALITY_SUMMARY_TABLE = table_name("mdo_phase1_quality_summary")


# ============================================================
# 5. MDO business config
# ============================================================

# Rule 7 / Rule 11 - Scenario Discount + No Discount Rule:
# Allow only PE-generated discount scenarios and keep 0% as the baseline scenario.
ALLOWED_SCENARIO_DISCOUNTS = parse_discount_list(
    get_param(
        "allowed_scenario_discounts",
        "0,0.05,0.075,0.10,0.15,0.20"
    )
)


# Rule 10 - Maximum Discount Rule:
# Do not recommend discount above this configured limit because business controls markdown depth.
MAX_DISCOUNT = get_float_param("max_discount", "0.20")


# Rule 16 - Elasticity Outlier Rule:
# Exclude scenarios where elasticity is too abnormal because unstable elasticity can create wrong recommendations.
MAX_ABS_ELASTICITY = 11.0

# Rule 50 - MDO Readiness Rule:
# Mark MDO as Ready only when valid scenario percentage is greater than or equal to this threshold.
MDO_READY_THRESHOLD = get_float_param("mdo_ready_threshold", "0.70")


# Rule 21 - Revenue Optimization Rule:
# Phase 1 selects the discount scenario with the highest expected revenue.
MDO_OBJECTIVE = get_param("mdo_objective", "revenue").lower()


# Rule 30 - Minimum History Rule:
# Exclude article-store combinations that do not have enough historical records for reliable recommendation.
MIN_HISTORY_DAYS = get_float_param("min_history_days", "30")


# Rule 31 - Price Variation Rule:
# Exclude rows where price movement is too low because elasticity cannot be trusted without price variation.
MIN_PRICE_VARIATION = get_float_param("min_price_variation", "0.0")


# Rule 32 - Discount Variation Rule:
# Exclude rows where discount movement is too low because markdown impact cannot be measured properly.
MIN_DISCOUNT_VARIATION = get_float_param("min_discount_variation", "0.01")


# Rule 33 - Low Confidence Prediction Rule:
# Exclude scenarios where model prediction confidence is below the configured threshold.
MIN_PREDICTION_CONFIDENCE = get_float_param("min_prediction_confidence", "0.60")


# Rule 35 - Sales Uplift Safety Rule:
# Exclude markdown scenarios that do not create positive expected quantity uplift.
MIN_SALES_UPLIFT_PCT = get_float_param("min_sales_uplift_pct", "0.00")


# ============================================================
# 6. Validation
# ============================================================

def validate_mdo_config() -> None:
    if not OUTPUT_SCHEMA:
        raise ValueError("output_schema cannot be empty")

    if not ALLOWED_SCENARIO_DISCOUNTS:
        raise ValueError("allowed_scenario_discounts cannot be empty")

    if 0.0 not in ALLOWED_SCENARIO_DISCOUNTS:
        raise ValueError("0% discount must be present as baseline scenario")

    if MAX_DISCOUNT < 0:
        raise ValueError("max_discount cannot be negative")

    if MAX_DISCOUNT > 1:
        raise ValueError("max_discount should be passed as decimal. Example: 0.20 for 20%")

    if MAX_ABS_ELASTICITY <= 0:
        raise ValueError("max_abs_elasticity must be greater than 0")

    if MDO_READY_THRESHOLD < 0 or MDO_READY_THRESHOLD > 1:
        raise ValueError("mdo_ready_threshold must be between 0 and 1")

    if MDO_OBJECTIVE not in ["revenue"]:
        raise ValueError("Phase 1 supports only mdo_objective = revenue")

    # Rule 30 - Minimum History Rule:
    # Minimum history must be zero or positive.
    if MIN_HISTORY_DAYS < 0:
        raise ValueError("min_history_days cannot be negative")

    # Rule 31 - Price Variation Rule:
    # Minimum price variation must be zero or positive.
    if MIN_PRICE_VARIATION < 0:
        raise ValueError("min_price_variation cannot be negative")

    # Rule 32 - Discount Variation Rule:
    # Minimum discount variation must be zero or positive.
    if MIN_DISCOUNT_VARIATION < 0:
        raise ValueError("min_discount_variation cannot be negative")

    # Rule 33 - Low Confidence Prediction Rule:
    # Prediction confidence threshold must be between 0 and 1.
    if MIN_PREDICTION_CONFIDENCE < 0 or MIN_PREDICTION_CONFIDENCE > 1:
        raise ValueError("min_prediction_confidence must be between 0 and 1")

    # Rule 35 - Sales Uplift Safety Rule:
    # Sales uplift threshold can be zero or positive.
    if MIN_SALES_UPLIFT_PCT < 0:
        raise ValueError("min_sales_uplift_pct cannot be negative")


def print_mdo_config() -> None:
    print("==================================================")
    print("MDO Phase 1 + Phase 2 Config")
    print("==================================================")
    print(f"OUTPUT_SCHEMA                      = {OUTPUT_SCHEMA}")
    print(f"TABLE_PREFIX                       = {TABLE_PREFIX}")
    print(f"PE_ENGINE_OUTPUT_TABLE             = {PE_ENGINE_OUTPUT_TABLE}")
    print(f"PREDICTIVE_SCENARIO_OUTPUT_TABLE   = {PREDICTIVE_SCENARIO_OUTPUT_TABLE}")
    print(f"BASE_DATA_QUALITY_SUMMARY_TABLE    = {BASE_DATA_QUALITY_SUMMARY_TABLE}")
    print(f"MDO_SCENARIO_RULE_OUTPUT_TABLE     = {MDO_SCENARIO_RULE_OUTPUT_TABLE}")
    print(f"MDO_FINAL_RECOMMENDATION_TABLE     = {MDO_FINAL_RECOMMENDATION_TABLE}")
    print(f"MDO_QUALITY_SUMMARY_TABLE          = {MDO_QUALITY_SUMMARY_TABLE}")
    print(f"ALLOWED_SCENARIO_DISCOUNTS         = {ALLOWED_SCENARIO_DISCOUNTS}")
    print(f"MAX_DISCOUNT                       = {MAX_DISCOUNT}")
    print(f"MAX_ABS_ELASTICITY                 = {MAX_ABS_ELASTICITY}")
    print(f"MDO_READY_THRESHOLD                = {MDO_READY_THRESHOLD}")
    print(f"MDO_OBJECTIVE                      = {MDO_OBJECTIVE}")
    print(f"MIN_HISTORY_DAYS                   = {MIN_HISTORY_DAYS}")
    print(f"MIN_PRICE_VARIATION                = {MIN_PRICE_VARIATION}")
    print(f"MIN_DISCOUNT_VARIATION             = {MIN_DISCOUNT_VARIATION}")
    print(f"MIN_PREDICTION_CONFIDENCE          = {MIN_PREDICTION_CONFIDENCE}")
    print(f"MIN_SALES_UPLIFT_PCT               = {MIN_SALES_UPLIFT_PCT}")
    print("==================================================")