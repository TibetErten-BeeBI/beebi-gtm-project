"""
MDO Phase 1 + Phase 2 quality summary.

Purpose:
- Track MDO data readiness.
- Track Include / Exclude / No Change counts.
- Track missing price, stock, inventory, invalid scenario counts.
- Track Phase 2 rule rejection counts.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


try:
    from .mdo_reason_codes import (
        ACTION_INCLUDE,
        ACTION_EXCLUDE,
        ACTION_NO_CHANGE,
        MISSING_PRICE,
        MISSING_STOCK,
        MISSING_INVENTORY,
        INVALID_SCENARIO,
        MISSING_PREDICTIVE_EXPECTED_QUANTITY,
        MISSING_PREDICTIVE_EXPECTED_REVENUE,
        MISSING_ELASTICITY,
        ELASTICITY_OUTLIER,
        MINIMUM_HISTORY_NOT_MET,
        LOW_PRICE_VARIATION,
        LOW_DISCOUNT_VARIATION,
        NEGATIVE_ELASTICITY,
        LOW_CONFIDENCE_PREDICTION,
        LOW_SALES_UPLIFT,
    )
except Exception:
    from mdo_reason_codes import (
        ACTION_INCLUDE,
        ACTION_EXCLUDE,
        ACTION_NO_CHANGE,
        MISSING_PRICE,
        MISSING_STOCK,
        MISSING_INVENTORY,
        INVALID_SCENARIO,
        MISSING_PREDICTIVE_EXPECTED_QUANTITY,
        MISSING_PREDICTIVE_EXPECTED_REVENUE,
        MISSING_ELASTICITY,
        ELASTICITY_OUTLIER,
        MINIMUM_HISTORY_NOT_MET,
        LOW_PRICE_VARIATION,
        LOW_DISCOUNT_VARIATION,
        NEGATIVE_ELASTICITY,
        LOW_CONFIDENCE_PREDICTION,
        LOW_SALES_UPLIFT,
    )


# ============================================================
# 1. Helper functions
# ============================================================

def sum_when(condition):
    """
    Count rows satisfying a Spark condition.
    """

    return F.sum(
        F.when(
            condition,
            F.lit(1),
        ).otherwise(
            F.lit(0)
        )
    )


def ensure_column(
    df: DataFrame,
    column_name: str,
    default_value,
) -> DataFrame:
    """
    Add a missing column with a default value.
    """

    if column_name not in df.columns:
        df = df.withColumn(
            column_name,
            F.lit(default_value),
        )

    return df


def ensure_required_summary_columns(
    df: DataFrame,
) -> DataFrame:
    """
    Ensure required columns exist before aggregation.
    """

    default_columns = {
        "valid_for_mdo_input": 0,
        "valid_for_price_elasticity": 0,
        "valid_price_flag": 0,
        "stock_available_flag": 0,
        "inventory_available_flag": 0,
        "scenario_valid_flag": 0,
        "mdo_valid_flag": 0,
        "scenario_discount": 0.0,
        "current_discount": 0.0,
        "scenario_type": "",
        "is_technical_baseline": 0,
        "is_current_baseline": 0,
        "is_candidate_scenario": 0,
        "action_flag": "",
        "reason_code": "",
    }

    for column_name, default_value in default_columns.items():
        df = ensure_column(
            df=df,
            column_name=column_name,
            default_value=default_value,
        )

    return df


def normalize_percentage_threshold(
    threshold_value: float,
) -> float:
    """
    Normalize readiness threshold to decimal format.

    Supported values:

    0.59 -> 0.59
    59   -> 0.59
    """

    # Threshold Rule - Convert the submitted threshold into a numeric value.
    try:
        normalized_threshold = float(
            threshold_value
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "MDO readiness threshold must be numeric. "
            f"Received: {threshold_value}"
        ) from error

    # Threshold Rule - Convert a whole percentage such as 59 into 0.59.
    if normalized_threshold > 1:
        normalized_threshold = (
            normalized_threshold / 100.0
        )

    # Threshold Rule - Allow only decimal values between 0 and 1.
    if not 0.0 <= normalized_threshold <= 1.0:
        raise ValueError(
            "MDO readiness threshold must be between "
            "0 and 1, or between 0 and 100. "
            f"Received: {threshold_value}"
        )

    # Threshold Rule - Store the normalized threshold with four decimal places.
    return round(
        normalized_threshold,
        4,
    )


# ============================================================
# 2. Scenario-level quality summary
# ============================================================

def build_scenario_quality_summary(
    scenario_rule_df: DataFrame,
) -> DataFrame:
    """
    Build quality metrics from candidate MDO scenarios.

    Readiness intentionally excludes:
    - the 0% technical PE baseline;
    - the current-discount business baseline.

    Only scenarios strictly above the current discount are included in the
    readiness denominator and candidate rejection counts.
    """

    scenario_rule_df = ensure_required_summary_columns(
        scenario_rule_df
    )

    # Derive scenario categories again for backward compatibility. This keeps
    # the summary correct even if an older scenario table does not contain all
    # three flag columns yet.
    scenario_rule_df = (
        scenario_rule_df
        .withColumn(
            "_summary_is_technical_baseline",
            F.when(
                (F.col("is_technical_baseline") == 1)
                | (F.abs(F.col("scenario_discount")) <= F.lit(0.0005)),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "_summary_is_current_baseline",
            F.when(
                (F.col("is_current_baseline") == 1)
                | (
                    F.abs(
                        F.col("scenario_discount")
                        - F.col("current_discount")
                    ) <= F.lit(0.0005)
                ),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "_summary_is_candidate",
            F.when(
                (F.col("is_candidate_scenario") == 1)
                | (F.upper(F.col("scenario_type")) == F.lit("CANDIDATE"))
                | (
                    F.col("scenario_discount")
                    > F.col("current_discount") + F.lit(0.0005)
                ),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
    )

    all_scenario_counts_df = scenario_rule_df.agg(
        F.count("*").alias("all_scenario_rows"),
        sum_when(
            F.col("_summary_is_technical_baseline") == 1
        ).alias("technical_baseline_rows"),
        sum_when(
            F.col("_summary_is_current_baseline") == 1
        ).alias("current_baseline_rows"),
        sum_when(
            F.col("_summary_is_candidate") == 1
        ).alias("candidate_scenario_rows"),
    ).fillna(0)

    candidate_df = scenario_rule_df.filter(
        F.col("_summary_is_candidate") == 1
    )

    # Keep the existing output field names. total_scenario_rows now means
    # total candidate rows used for MDO readiness.
    candidate_summary_df = candidate_df.agg(
        F.count("*").alias("total_scenario_rows"),

        sum_when(
            F.col("valid_for_mdo_input") == 1
        ).alias("valid_for_mdo_input_rows"),

        sum_when(
            F.col("valid_for_price_elasticity") == 1
        ).alias("valid_for_price_elasticity_rows"),

        sum_when(
            F.col("valid_price_flag") == 1
        ).alias("valid_price_rows"),

        sum_when(
            F.col("stock_available_flag") == 1
        ).alias("stock_available_rows"),

        sum_when(
            F.col("inventory_available_flag") == 1
        ).alias("inventory_available_rows"),

        sum_when(
            F.col("scenario_valid_flag") == 1
        ).alias("valid_scenario_rows"),

        sum_when(
            F.col("scenario_valid_flag") == 0
        ).alias("invalid_scenario_rows"),

        sum_when(
            F.col("reason_code") == MISSING_PRICE
        ).alias("missing_price_count"),

        sum_when(
            F.col("reason_code") == MISSING_STOCK
        ).alias("missing_stock_count"),

        sum_when(
            F.col("reason_code") == MISSING_INVENTORY
        ).alias("missing_inventory_count"),

        sum_when(
            F.col("reason_code") == INVALID_SCENARIO
        ).alias("invalid_scenario_count"),

        sum_when(
            F.col("reason_code")
            == MISSING_PREDICTIVE_EXPECTED_QUANTITY
        ).alias("missing_predictive_expected_quantity_count"),

        sum_when(
            F.col("reason_code")
            == MISSING_PREDICTIVE_EXPECTED_REVENUE
        ).alias("missing_predictive_expected_revenue_count"),

        sum_when(
            F.col("reason_code") == MISSING_ELASTICITY
        ).alias("missing_elasticity_count"),

        sum_when(
            F.col("reason_code") == ELASTICITY_OUTLIER
        ).alias("elasticity_outlier_count"),

        sum_when(
            F.col("reason_code") == MINIMUM_HISTORY_NOT_MET
        ).alias("minimum_history_not_met_count"),

        sum_when(
            F.col("reason_code") == LOW_PRICE_VARIATION
        ).alias("low_price_variation_count"),

        sum_when(
            F.col("reason_code") == LOW_DISCOUNT_VARIATION
        ).alias("low_discount_variation_count"),

        sum_when(
            F.col("reason_code") == NEGATIVE_ELASTICITY
        ).alias("negative_elasticity_count"),

        sum_when(
            F.col("reason_code") == LOW_CONFIDENCE_PREDICTION
        ).alias("low_confidence_prediction_count"),

        sum_when(
            F.col("reason_code") == LOW_SALES_UPLIFT
        ).alias("low_sales_uplift_count"),
    ).fillna(0)

    summary_df = (
        candidate_summary_df
        .crossJoin(all_scenario_counts_df)
        .withColumn(
            "mdo_ready_pct",
            F.when(
                F.col("total_scenario_rows") > 0,
                F.round(
                    F.col("valid_scenario_rows")
                    / F.col("total_scenario_rows"),
                    4,
                ),
            ).otherwise(F.lit(0.0)),
        )
    )

    return summary_df


# ============================================================
# 3. Final recommendation summary
# ============================================================

def build_final_recommendation_summary(
    final_recommendation_df: DataFrame,
) -> DataFrame:
    """
    Build metrics from final MDO recommendations.
    """

    final_recommendation_df = (
        ensure_required_summary_columns(
            final_recommendation_df
        )
    )

    summary_df = final_recommendation_df.agg(
        F.count("*").alias(
            "total_final_recommendation_rows"
        ),

        # Recommendation Rule - Count final Include recommendations.
        sum_when(
            F.col("action_flag")
            == ACTION_INCLUDE
        ).alias(
            "include_count"
        ),

        # Recommendation Rule - Count final Exclude recommendations.
        sum_when(
            F.col("action_flag")
            == ACTION_EXCLUDE
        ).alias(
            "exclude_count"
        ),

        # Recommendation Rule - Count final No Change recommendations.
        sum_when(
            F.col("action_flag")
            == ACTION_NO_CHANGE
        ).alias(
            "no_change_count"
        ),

        # Markdown Rule - Count recommendations where discount is increased.
        sum_when(
            F.col("recommended_discount")
            > F.col("current_discount")
        ).alias(
            "markdown_change_recommendation_count"
        ),
    )

    return summary_df


# ============================================================
# 4. Final MDO quality summary
# ============================================================

def build_mdo_quality_summary(
    scenario_rule_df: DataFrame,
    final_recommendation_df: DataFrame,
    mdo_ready_threshold: float,
) -> DataFrame:
    """
    Combine scenario and recommendation summaries.

    Both readiness and threshold are stored as decimal values.

    Example:

    readiness = 0.6479
    threshold = 0.5900
    status    = Ready
    """

    # Summary Rule - Create one aggregated row containing scenario quality metrics.
    scenario_summary_df = (
        build_scenario_quality_summary(
            scenario_rule_df
        )
    )

    # Summary Rule - Create one aggregated row containing recommendation metrics.
    final_summary_df = (
        build_final_recommendation_summary(
            final_recommendation_df
        )
    )

    # Threshold Rule - Normalize both 0.59 and 59 into the decimal value 0.59.
    normalized_threshold = (
        normalize_percentage_threshold(
            mdo_ready_threshold
        )
    )

    quality_summary_df = (
        scenario_summary_df

        # Summary Rule - Combine scenario metrics and recommendation metrics into one row.
        .crossJoin(
            final_summary_df
        )

        # Threshold Rule - Store the normalized readiness threshold as a decimal.
        .withColumn(
            "mdo_ready_threshold",
            F.lit(
                normalized_threshold
            ).cast("double"),
        )

        # Rule 50 - Mark the run Ready when readiness is greater than or equal to the threshold.
        .withColumn(
            "mdo_readiness_status",
            F.when(
                F.col("mdo_ready_pct")
                >= F.col(
                    "mdo_ready_threshold"
                ),
                F.lit("Ready"),
            ).otherwise(
                F.lit("Not Ready")
            ),
        )

        # Audit Rule - Record when the MDO quality summary was generated.
        .withColumn(
            "created_timestamp",
            F.current_timestamp(),
        )
    )

    print(
        "========================================"
    )
    print(
        "MDO READINESS VALIDATION"
    )
    print(
        f"Submitted threshold: "
        f"{mdo_ready_threshold}"
    )
    print(
        f"Normalized threshold: "
        f"{normalized_threshold}"
    )

    # Validation Rule - Display readiness values used to determine the final status.
    quality_summary_df.select(
        "total_scenario_rows",
        "valid_scenario_rows",
        "invalid_scenario_rows",
        "mdo_ready_pct",
        "mdo_ready_threshold",
        "mdo_readiness_status",
    ).show(
        truncate=False
    )

    return quality_summary_df


# ============================================================
# 5. Write summary table
# ============================================================

def write_mdo_quality_summary(
    quality_summary_df: DataFrame,
    output_table_name: str,
) -> None:
    """
    Write the latest MDO quality summary as a Delta table.
    """

    # Output Rule - Replace the existing summary table with the latest run results.
    (
        quality_summary_df
        .write
        .format("delta")
        .mode("overwrite")
        .option(
            "overwriteSchema",
            "true",
        )
        .saveAsTable(
            output_table_name
        )
    )

    print(
        "MDO quality summary written successfully: "
        f"{output_table_name}"
    )