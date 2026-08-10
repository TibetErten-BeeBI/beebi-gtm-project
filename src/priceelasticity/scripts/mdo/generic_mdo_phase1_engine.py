"""
Generic MDO Phase 1 + Phase 2 Engine.

This is the main runnable file for MDO rules.

It will:
1. Read PE / predictive scenario output
2. Apply Phase 1 + Phase 2 MDO rules
3. Write scenario-level rule output
4. Write final MDO recommendation output
5. Write MDO quality summary
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. Fix import path for Databricks Python file execution
# ============================================================

import inspect

try:
    SCRIPT_PATH = __file__
except NameError:
    SCRIPT_PATH = inspect.getfile(inspect.currentframe())

CURRENT_DIR = os.path.dirname(os.path.abspath(SCRIPT_PATH))

if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)

sys.path.insert(0, CURRENT_DIR)

print(f"MDO script directory added to path: {CURRENT_DIR}")


from mdo_config import (
    validate_mdo_config,
    print_mdo_config,
    PE_ENGINE_OUTPUT_TABLE,
    PREDICTIVE_SCENARIO_OUTPUT_TABLE,
    MDO_SCENARIO_RULE_OUTPUT_TABLE,
    MDO_FINAL_RECOMMENDATION_TABLE,
    MDO_QUALITY_SUMMARY_TABLE,
    ALLOWED_SCENARIO_DISCOUNTS,
    MAX_DISCOUNT,
    MAX_ABS_ELASTICITY,
    MDO_READY_THRESHOLD,
    MIN_HISTORY_DAYS,
    MIN_PRICE_VARIATION,
    MIN_DISCOUNT_VARIATION,
    MIN_PREDICTION_CONFIDENCE,
    MIN_SALES_UPLIFT_PCT,
)

print(f"Loaded mdo_config MAX_ABS_ELASTICITY = {MAX_ABS_ELASTICITY}")
print(f"Loaded mdo_config MIN_PRICE_VARIATION = {MIN_PRICE_VARIATION}")

from mdo_rule_engine import run_mdo_phase1_rules
from mdo_quality_summary import (
    build_mdo_quality_summary,
    write_mdo_quality_summary,
)


# ============================================================
# 2. Spark session
# ============================================================

def get_spark() -> SparkSession:
    return SparkSession.builder.appName("generic_mdo_phase1_phase2_engine").getOrCreate()


# ============================================================
# 3. Helpers
# ============================================================

def table_exists(spark: SparkSession, table_name: str) -> bool:
    try:
        spark.table(table_name).limit(1).count()
        return True
    except Exception:
        return False


def read_table_if_exists(spark: SparkSession, table_name: str):
    if table_exists(spark, table_name):
        print(f"Reading table: {table_name}")
        return spark.table(table_name)

    print(f"Table not found: {table_name}")
    return None


def print_table_count(df, table_label: str) -> None:
    if df is None:
        print(f"{table_label}: DataFrame is None")
        return

    row_count = df.count()
    print(f"{table_label} row count: {row_count}")


def write_delta_table(df, table_name: str) -> None:
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )

    print(f"Written table successfully: {table_name}")


def normalize_discount_column(df, column_name: str):
    return (
        df.withColumn(
            column_name,
            F.when(F.col(column_name).isNull(), F.lit(None).cast("double"))
            .when(F.col(column_name) > 1, F.col(column_name) / F.lit(100.0))
            .otherwise(F.col(column_name).cast("double"))
        )
    )


# ============================================================
# 4. Prepare MDO source dataframe
# ============================================================

def prepare_mdo_source_df(spark: SparkSession):
    """
    MDO needs scenario output plus PE elasticity output.

    Preferred source:
    - pe_predictive_scenario_output

    Optional enrichment:
    - pe_price_elasticity_engine_output_dev

    If both are available, we join useful PE columns into predictive scenario output.
    """

    # Rule 17 / Rule 18 - Prediction + Revenue Availability Rule:
    # Read predictive scenario output because MDO needs expected quantity and expected revenue for each discount.
    predictive_df = read_table_if_exists(
        spark,
        PREDICTIVE_SCENARIO_OUTPUT_TABLE,
    )

    # Rule 14 / Rule 29 - Elasticity Validity + PE Confidence Rule:
    # Read PE engine output because MDO needs elasticity, lift, and PE validity fields.
    pe_engine_df = read_table_if_exists(
        spark,
        PE_ENGINE_OUTPUT_TABLE,
    )

    if predictive_df is None and pe_engine_df is None:
        raise ValueError(
            "Both predictive scenario output and PE engine output tables are missing. "
            "Run PE pipeline before MDO."
        )

    if predictive_df is None:
        print("Predictive scenario output missing. Using PE engine output directly.")
        return pe_engine_df

    if pe_engine_df is None:
        print("PE engine output missing. Using predictive scenario output directly.")
        return predictive_df

    print("Both predictive scenario output and PE engine output found.")
    print("Using predictive scenario output as main source and enriching from PE engine output.")

    # Rule 7 - Scenario Discount Rule:
    # Normalize scenario discount so both 10 and 0.10 are treated as 10%.
    if "scenario_discount" in predictive_df.columns:
        predictive_df = normalize_discount_column(predictive_df, "scenario_discount")

    if "scenario_discount_pct" in predictive_df.columns and "scenario_discount" not in predictive_df.columns:
        predictive_df = predictive_df.withColumnRenamed(
            "scenario_discount_pct",
            "scenario_discount"
        )
        predictive_df = normalize_discount_column(predictive_df, "scenario_discount")

    if "scenario_discount" in pe_engine_df.columns:
        pe_engine_df = normalize_discount_column(pe_engine_df, "scenario_discount")

    if "scenario_discount_pct" in pe_engine_df.columns and "scenario_discount" not in pe_engine_df.columns:
        pe_engine_df = pe_engine_df.withColumnRenamed(
            "scenario_discount_pct",
            "scenario_discount"
        )
        pe_engine_df = normalize_discount_column(pe_engine_df, "scenario_discount")

    join_columns = []

    # Rule 6 - Product-Store-Date Grain Rule:
    # Join PE and scenario outputs at article-store-date-scenario grain where common columns exist.
    for column_name in ["pe_article", "pe_store_group", "date", "scenario_discount"]:
        if column_name in predictive_df.columns and column_name in pe_engine_df.columns:
            join_columns.append(column_name)

    if len(join_columns) < 3:
        print("Not enough common join columns between predictive and PE engine output.")
        print("Using predictive scenario output directly.")
        return predictive_df

    # MDO Enrichment:
    # Add useful PE fields into scenario output so MDO rules have all required inputs.
    pe_columns_to_add = [
        "expected_quantity_lift_pct",
        "expected_revenue_lift_pct",
        "expected_quantity_elasticity",
        "valid_for_price_elasticity",
        "valid_for_mdo_input",
        "valid_price_flag",
        "stock_available_flag",
        "inventory_available_flag",
        "pe_unit_price",
        "pe_quantity",
        "pe_store_stock_quantity",
        "pe_inventory_onhand_quantity",

        # Rule 30 - Minimum History Rule:
        # Add history_days so MDO can exclude article-store rows with insufficient sales history.
        "history_days",

        # Rule 31 - Price Variation Rule:
        # Add price_variation so MDO can exclude rows where price movement is too low.
        "price_variation",

        # Rule 32 - Discount Variation Rule:
        # Add discount_variation so MDO can exclude rows where discount movement is too low.
        "discount_variation",

        # Rule 33 - Low Confidence Prediction Rule:
        # Add prediction_confidence so MDO can exclude low-confidence model predictions.
        "prediction_confidence",
    ]

    selected_pe_columns = join_columns.copy()

    for column_name in pe_columns_to_add:
        if column_name in pe_engine_df.columns and column_name not in selected_pe_columns:
            selected_pe_columns.append(column_name)

    # Rule 6 - Product-Store-Date Grain Rule:
    # Drop duplicates before join to avoid creating duplicate scenario rows.
    pe_engine_selected_df = pe_engine_df.select(*selected_pe_columns).dropDuplicates(join_columns)

    duplicate_columns = [
        column_name
        for column_name in selected_pe_columns
        if column_name not in join_columns and column_name in predictive_df.columns
    ]

    for column_name in duplicate_columns:
        pe_engine_selected_df = pe_engine_selected_df.withColumnRenamed(
            column_name,
            f"pe_engine_{column_name}"
        )

    joined_df = predictive_df.join(
        pe_engine_selected_df,
        on=join_columns,
        how="left",
    )

    # MDO Enrichment:
    # Prefer existing predictive columns; use PE engine columns only when predictive values are missing.
    for column_name in pe_columns_to_add:
        pe_engine_column = f"pe_engine_{column_name}"

        if column_name in joined_df.columns and pe_engine_column in joined_df.columns:
            joined_df = joined_df.withColumn(
                column_name,
                F.coalesce(F.col(column_name), F.col(pe_engine_column))
            ).drop(pe_engine_column)

        elif column_name not in joined_df.columns and pe_engine_column in joined_df.columns:
            joined_df = joined_df.withColumnRenamed(pe_engine_column, column_name)

    return joined_df


# ============================================================
# 5. Main MDO execution
# ============================================================

def main() -> None:
    print("==================================================")
    print("Starting Generic MDO Phase 1 + Phase 2 Engine")
    print("==================================================")

    # MDO Config Validation:
    # Validate Phase 1 and Phase 2 thresholds before running.
    validate_mdo_config()

    # MDO Config Logging:
    # Print all input tables, output tables, and business settings for debugging.
    print_mdo_config()

    spark = get_spark()

    # MDO Input Preparation:
    # Read PE outputs and prepare one source dataframe for MDO rules.
    source_df = prepare_mdo_source_df(spark)

    print_table_count(source_df, "MDO source dataframe")

    print("Applying MDO Phase 1 + Phase 2 rules...")

    # Phase 1 + Phase 2 Rule Execution:
    # Apply MDO rules and produce scenario-level output plus final recommendation output.
    scenario_rule_df, final_recommendation_df = run_mdo_phase1_rules(
        source_df=source_df,
        allowed_scenario_discounts=ALLOWED_SCENARIO_DISCOUNTS,
        max_discount=MAX_DISCOUNT,
        max_abs_elasticity=MAX_ABS_ELASTICITY,
        min_history_days=MIN_HISTORY_DAYS,
        min_price_variation=MIN_PRICE_VARIATION,
        min_discount_variation=MIN_DISCOUNT_VARIATION,
        min_prediction_confidence=MIN_PREDICTION_CONFIDENCE,
        min_sales_uplift_pct=MIN_SALES_UPLIFT_PCT,
    )

    print_table_count(scenario_rule_df, "MDO scenario rule output")
    print_table_count(final_recommendation_df, "MDO final recommendation output")

    print("Writing MDO scenario-level rule output...")

    write_delta_table(
        scenario_rule_df,
        MDO_SCENARIO_RULE_OUTPUT_TABLE,
    )

    print("Writing MDO final recommendation output...")

    write_delta_table(
        final_recommendation_df,
        MDO_FINAL_RECOMMENDATION_TABLE,
    )

    print("Building MDO quality summary...")
    print(
        "Readiness excludes the 0% baseline and uses only non-zero "
        "recommendation candidate scenarios."
    )

    quality_summary_df = build_mdo_quality_summary(
        scenario_rule_df=scenario_rule_df,
        final_recommendation_df=final_recommendation_df,
        mdo_ready_threshold=MDO_READY_THRESHOLD,
    )

    write_mdo_quality_summary(
        quality_summary_df=quality_summary_df,
        output_table_name=MDO_QUALITY_SUMMARY_TABLE,
    )

    print("==================================================")
    print("Generic MDO Phase 1 + Phase 2 Engine completed successfully")
    print("==================================================")


if __name__ == "__main__":
    main()