"""
MDO Phase 1 + Phase 2 rule engine.

This file applies the main MDO rules:
- input eligibility
- price validation
- stock validation
- inventory validation
- minimum history validation
- price variation validation
- discount variation validation
- prediction confidence validation
- negative elasticity validation
- sales uplift safety validation
- scenario discount validation
- scenario price validation
- prediction validation
- elasticity validation
- stock cap
- baseline comparison
- best scenario selection
- Include / Exclude / No Change action flag
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


try:
    from .mdo_reason_codes import (
        MISSING_PRICE,
        MISSING_STOCK,
        MISSING_INVENTORY,
        INVALID_MDO_INPUT,
        INVALID_PRICE_ELASTICITY_OUTPUT,
        NEGATIVE_QUANTITY,
        MINIMUM_HISTORY_NOT_MET,
        LOW_PRICE_VARIATION,
        LOW_DISCOUNT_VARIATION,
        NEGATIVE_ELASTICITY,
        LOW_CONFIDENCE_PREDICTION,
        LOW_SALES_UPLIFT,
        INVALID_SCENARIO_DISCOUNT,
        INVALID_SCENARIO_PRICE,
        SCENARIO_DISCOUNT_BELOW_CURRENT_DISCOUNT,
        SCENARIO_DISCOUNT_ABOVE_MAX_DISCOUNT,
        MISSING_PREDICTIVE_EXPECTED_QUANTITY,
        MISSING_PREDICTIVE_EXPECTED_REVENUE,
        MISSING_ELASTICITY,
        ELASTICITY_OUTLIER,
        INVALID_SCENARIO,
        BASELINE_SCENARIO_IS_BEST,
        REVENUE_IMPROVES_WITH_MARKDOWN,
        NO_REVENUE_IMPROVEMENT,
        NO_VALID_SCENARIO_FOUND,
        ACTION_INCLUDE,
        ACTION_EXCLUDE,
        ACTION_NO_CHANGE,
        REASON_PRIORITY,
    )
except Exception:
    from mdo_reason_codes import (
        MISSING_PRICE,
        MISSING_STOCK,
        MISSING_INVENTORY,
        INVALID_MDO_INPUT,
        INVALID_PRICE_ELASTICITY_OUTPUT,
        NEGATIVE_QUANTITY,
        MINIMUM_HISTORY_NOT_MET,
        LOW_PRICE_VARIATION,
        LOW_DISCOUNT_VARIATION,
        NEGATIVE_ELASTICITY,
        LOW_CONFIDENCE_PREDICTION,
        LOW_SALES_UPLIFT,
        INVALID_SCENARIO_DISCOUNT,
        INVALID_SCENARIO_PRICE,
        SCENARIO_DISCOUNT_BELOW_CURRENT_DISCOUNT,
        SCENARIO_DISCOUNT_ABOVE_MAX_DISCOUNT,
        MISSING_PREDICTIVE_EXPECTED_QUANTITY,
        MISSING_PREDICTIVE_EXPECTED_REVENUE,
        MISSING_ELASTICITY,
        ELASTICITY_OUTLIER,
        INVALID_SCENARIO,
        BASELINE_SCENARIO_IS_BEST,
        REVENUE_IMPROVES_WITH_MARKDOWN,
        NO_REVENUE_IMPROVEMENT,
        NO_VALID_SCENARIO_FOUND,
        ACTION_INCLUDE,
        ACTION_EXCLUDE,
        ACTION_NO_CHANGE,
        REASON_PRIORITY,
    )


# ============================================================
# 1. Common helpers
# ============================================================

# Rule 6 - Product-Store-Date Grain Rule:
# MDO recommendations are created at article-store-date level.
GRAIN_COLUMNS = ["pe_article", "pe_store_group", "date"]


def existing_columns(df: DataFrame, candidates: list) -> list:
    return [column_name for column_name in candidates if column_name in df.columns]


def coalesce_existing_columns(
    df: DataFrame,
    candidates: list,
    default_value=None,
    data_type: str = None,
):
    cols = []

    for column_name in candidates:
        if column_name in df.columns:
            col_expr = F.col(column_name)
            if data_type:
                col_expr = col_expr.cast(data_type)
            cols.append(col_expr)

    if not cols:
        if data_type:
            return F.lit(default_value).cast(data_type)
        return F.lit(default_value)

    return F.coalesce(*cols)


def normalize_discount(discount_col):
    """
    Handles both decimal and percentage discount formats.

    Example:
    0.10 stays 0.10
    10 becomes 0.10
    """
    return (
        F.when(discount_col.isNull(), F.lit(None).cast("double"))
        .when(discount_col > 1, discount_col / F.lit(100.0))
        .otherwise(discount_col)
    )


def add_reason_priority(df: DataFrame) -> DataFrame:
    """
    Adds numeric priority for reason_code.
    Lower value means higher priority.
    """

    mapping_expr_items = []

    for reason_code, priority in REASON_PRIORITY.items():
        mapping_expr_items.append(F.lit(reason_code))
        mapping_expr_items.append(F.lit(priority))

    reason_priority_map = F.create_map(*mapping_expr_items)

    # Rule 28 - Reason Priority Rule:
    # If multiple rules fail, assign priority so the main failure reason can be shown first.
    return df.withColumn(
        "reason_priority",
        F.coalesce(
            reason_priority_map[F.col("reason_code")],
            F.lit(999)
        )
    )


# ============================================================
# 2. Standardize columns
# ============================================================

def standardize_mdo_input_columns(df: DataFrame) -> DataFrame:
    """
    Standardizes current PE output columns into names used by MDO.

    This keeps MDO generic even if PE output uses:
    product/store OR pe_article/pe_store_group.
    """

    df = (
        df
        .withColumn(
            "pe_article",
            coalesce_existing_columns(
                df,
                ["pe_article", "article", "product", "group_article_id"],
                None,
                "string",
            )
        )
        .withColumn(
            "pe_store_group",
            coalesce_existing_columns(
                df,
                ["pe_store_group", "store", "store_group", "reporting_unit"],
                None,
                "string",
            )
        )
        .withColumn(
            "date",
            F.to_date(
                coalesce_existing_columns(
                    df,
                    ["date", "pe_date", "business_date", "snapshot_date"],
                    None,
                    "string",
                )
            )
        )
        .withColumn(
            "base_unit_price",
            coalesce_existing_columns(
                df,
                ["base_unit_price", "pe_unit_price", "unit_price", "actual_unit_price"],
                None,
                "double",
            )
        )
        .withColumn(
            "pe_unit_price",
            coalesce_existing_columns(
                df,
                ["pe_unit_price", "base_unit_price", "unit_price", "actual_unit_price"],
                None,
                "double",
            )
        )
        .withColumn(
            "pe_quantity",
            coalesce_existing_columns(
                df,
                ["pe_quantity", "actual_quantity", "quantity", "sales_quantity"],
                None,
                "double",
            )
        )
        .withColumn(
            "scenario_discount_raw",
            coalesce_existing_columns(
                df,
                ["scenario_discount", "scenario_discount_pct"],
                None,
                "double",
            )
        )
        .withColumn(
            "scenario_discount",
            normalize_discount(F.col("scenario_discount_raw"))
        )
        .withColumn(
            "scenario_unit_price",
            coalesce_existing_columns(
                df,
                ["scenario_unit_price", "scenario_price", "discounted_price"],
                None,
                "double",
            )
        )
        .withColumn(
            "current_discount_raw",
            coalesce_existing_columns(
                df,
                ["current_discount", "current_discount_pct"],
                0.0,
                "double",
            )
        )
        .withColumn(
            "current_discount",
            normalize_discount(F.col("current_discount_raw"))
        )
        .withColumn(
            "predictive_expected_quantity",
            coalesce_existing_columns(
                df,
                [
                    "predictive_expected_quantity",
                    "expected_quantity",
                    "predictive_scenario_quantity",
                ],
                None,
                "double",
            )
        )
        .withColumn(
            "predictive_expected_revenue",
            coalesce_existing_columns(
                df,
                [
                    "predictive_expected_revenue",
                    "expected_revenue",
                    "predictive_scenario_revenue",
                ],
                None,
                "double",
            )
        )
        .withColumn(
            "expected_quantity_lift_pct",
            coalesce_existing_columns(
                df,
                ["expected_quantity_lift_pct", "quantity_lift_pct"],
                None,
                "double",
            )
        )
        .withColumn(
            "expected_revenue_lift_pct",
            coalesce_existing_columns(
                df,
                ["expected_revenue_lift_pct", "revenue_lift_pct"],
                None,
                "double",
            )
        )
        .withColumn(
            "expected_quantity_elasticity",
            coalesce_existing_columns(
                df,
                ["expected_quantity_elasticity", "price_elasticity", "elasticity"],
                None,
                "double",
            )
        )
        .withColumn(
            "pe_store_stock_quantity",
            coalesce_existing_columns(
                df,
                [
                    "pe_store_stock_quantity",
                    "store_stock_quantity",
                    "available_stock",
                    "stock_quantity",
                    "on_hand_stock",
                ],
                None,
                "double",
            )
        )
        .withColumn(
            "pe_inventory_onhand_quantity",
            coalesce_existing_columns(
                df,
                [
                    "pe_inventory_onhand_quantity",
                    "inventory_onhand_quantity",
                    "inventory_quantity",
                    "onhand_quantity",
                ],
                None,
                "double",
            )
        )

        # Rule 30 - Minimum History Rule:
        # Standardize history days so MDO can exclude article-store rows with insufficient history.
        .withColumn(
            "history_days",
            coalesce_existing_columns(
                df,
                ["history_days", "history_count", "sales_history_days", "available_history_days"],
                None,
                "double",
            )
        )

        # Rule 31 - Price Variation Rule:
        # Standardize price variation so MDO can exclude rows where price movement is too low.
        .withColumn(
            "price_variation",
            coalesce_existing_columns(
                df,
                ["price_variation", "price_stddev", "unit_price_variation"],
                None,
                "double",
            )
        )

        # Rule 32 - Discount Variation Rule:
        # Standardize discount variation so MDO can exclude rows where discount movement is too low.
        .withColumn(
            "discount_variation",
            coalesce_existing_columns(
                df,
                ["discount_variation", "discount_stddev", "discount_pct_variation"],
                None,
                "double",
            )
        )

        # Rule 33 - Low Confidence Prediction Rule:
        # Standardize prediction confidence so MDO can avoid low-confidence model outputs.
        .withColumn(
            "prediction_confidence",
            coalesce_existing_columns(
                df,
                ["prediction_confidence", "model_confidence", "confidence_score"],
                None,
                "double",
            )
        )
    )

    if "valid_price_flag" in df.columns:
        df = df.withColumn(
            "valid_price_flag",
            F.coalesce(F.col("valid_price_flag").cast("int"), F.lit(0))
        )
    else:
        df = df.withColumn(
            "valid_price_flag",
            F.when(F.col("pe_unit_price") > 0, F.lit(1)).otherwise(F.lit(0))
        )

    if "stock_available_flag" in df.columns:
        df = df.withColumn(
            "stock_available_flag",
            F.coalesce(F.col("stock_available_flag").cast("int"), F.lit(0))
        )
    else:
        df = df.withColumn(
            "stock_available_flag",
            F.when(F.col("pe_store_stock_quantity") > 0, F.lit(1)).otherwise(F.lit(0))
        )

    if "inventory_available_flag" in df.columns:
        df = df.withColumn(
            "inventory_available_flag",
            F.coalesce(F.col("inventory_available_flag").cast("int"), F.lit(0))
        )
    else:
        df = df.withColumn(
            "inventory_available_flag",
            F.when(F.col("pe_inventory_onhand_quantity") > 0, F.lit(1)).otherwise(F.lit(0))
        )

    if "valid_for_price_elasticity" in df.columns:
        df = df.withColumn(
            "valid_for_price_elasticity",
            F.coalesce(F.col("valid_for_price_elasticity").cast("int"), F.lit(0))
        )
    else:
        df = df.withColumn("valid_for_price_elasticity", F.lit(1))

    if "valid_for_mdo_input" in df.columns:
        df = df.withColumn(
            "valid_for_mdo_input",
            F.coalesce(F.col("valid_for_mdo_input").cast("int"), F.lit(0))
        )
    else:
        df = df.withColumn(
            "valid_for_mdo_input",
            F.when(
                (F.col("valid_price_flag") == 1)
                & (F.col("stock_available_flag") == 1)
                & (F.col("inventory_available_flag") == 1),
                F.lit(1),
            ).otherwise(F.lit(0))
        )

    return df


# ============================================================
# 3. Apply scenario-level MDO rules
# ============================================================

def apply_mdo_phase1_scenario_rules(
    source_df: DataFrame,
    allowed_scenario_discounts: list,
    max_discount: float,
    max_abs_elasticity: float,
    min_history_days: float,
    min_price_variation: float,
    min_discount_variation: float,
    min_prediction_confidence: float,
    min_sales_uplift_pct: float,
) -> DataFrame:
    """
    Apply scenario-level MDO validation rules.

    Important baseline behavior:
    - 0% is retained as a technical PE baseline.
    - The current discount is the MDO business baseline.
    - Only scenarios strictly above the current discount are candidates.
    - Candidate quantity/revenue lifts are calculated against the current
      scenario before candidate validation is applied.
    """

    allowed_discount_values = [
        round(float(discount), 3)
        for discount in allowed_scenario_discounts
    ]

    df = standardize_mdo_input_columns(source_df)

    df = (
        df
        .withColumn(
            "scenario_discount_rounded",
            F.round(F.col("scenario_discount"), 3),
        )
        .withColumn(
            "current_discount_rounded",
            F.round(F.col("current_discount"), 3),
        )
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
                    - F.col("current_discount")
                ) <= F.lit(0.0005),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "is_candidate_scenario",
            F.when(
                F.col("scenario_discount")
                > F.col("current_discount") + F.lit(0.0005),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "is_baseline_scenario",
            F.col("is_current_baseline"),
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
    )

    # Rule 19 - Stock Cap Rule.
    df = df.withColumn(
        "available_stock_for_cap",
        F.least(
            F.col("pe_store_stock_quantity"),
            F.col("pe_inventory_onhand_quantity"),
        ),
    )

    df = df.withColumn(
        "capped_expected_quantity",
        F.when(
            F.col("predictive_expected_quantity").isNotNull()
            & F.col("available_stock_for_cap").isNotNull(),
            F.least(
                F.col("predictive_expected_quantity"),
                F.col("available_stock_for_cap"),
            ),
        ).otherwise(F.col("predictive_expected_quantity")),
    )

    df = df.withColumn(
        "capped_expected_revenue",
        F.when(
            F.col("capped_expected_quantity").isNotNull()
            & F.col("scenario_unit_price").isNotNull(),
            F.col("capped_expected_quantity")
            * F.col("scenario_unit_price"),
        ).otherwise(F.col("predictive_expected_revenue")),
    )

    # Build the current-discount baseline before applying candidate uplift
    # rules. This lets MDO validate candidates against the actual current
    # business position rather than against the technical 0% PE row.
    current_baseline_window = Window.partitionBy(*GRAIN_COLUMNS).orderBy(
        F.col("capped_expected_revenue").desc_nulls_last(),
        F.col("scenario_unit_price").desc_nulls_last(),
    )

    current_baseline_df = (
        df
        .filter(F.col("is_current_baseline") == 1)
        .withColumn(
            "_current_baseline_rank",
            F.row_number().over(current_baseline_window),
        )
        .filter(F.col("_current_baseline_rank") == 1)
        .select(
            *GRAIN_COLUMNS,
            F.col("scenario_discount").alias(
                "current_baseline_discount"
            ),
            F.col("scenario_unit_price").alias(
                "current_baseline_unit_price"
            ),
            F.col("capped_expected_quantity").alias(
                "current_baseline_expected_quantity"
            ),
            F.col("capped_expected_revenue").alias(
                "current_baseline_expected_revenue"
            ),
        )
    )

    df = (
        df
        .join(
            current_baseline_df,
            on=GRAIN_COLUMNS,
            how="left",
        )
        .withColumn(
            "mdo_expected_quantity_lift_pct",
            F.when(
                F.col("current_baseline_expected_quantity") > 0,
                (
                    F.col("capped_expected_quantity")
                    - F.col("current_baseline_expected_quantity")
                )
                / F.col("current_baseline_expected_quantity"),
            )
            .when(
                F.col("is_current_baseline") == 1,
                F.lit(0.0),
            ),
        )
        .withColumn(
            "mdo_expected_revenue_lift_pct",
            F.when(
                F.col("current_baseline_expected_revenue") > 0,
                (
                    F.col("capped_expected_revenue")
                    - F.col("current_baseline_expected_revenue")
                )
                / F.col("current_baseline_expected_revenue"),
            )
            .when(
                F.col("is_current_baseline") == 1,
                F.lit(0.0),
            ),
        )
    )

    reason_expr = (
        # Basic data-quality checks apply to every row, including the current
        # baseline. Candidate-specific checks are added afterwards.
        F.when(
            (F.col("valid_price_flag") != 1)
            | F.col("pe_unit_price").isNull()
            | (F.col("pe_unit_price") <= 0),
            F.lit(MISSING_PRICE),
        )
        .when(
            (F.col("stock_available_flag") != 1)
            | F.col("pe_store_stock_quantity").isNull()
            | (F.col("pe_store_stock_quantity") <= 0),
            F.lit(MISSING_STOCK),
        )
        .when(
            (F.col("inventory_available_flag") != 1)
            | F.col("pe_inventory_onhand_quantity").isNull()
            | (F.col("pe_inventory_onhand_quantity") <= 0),
            F.lit(MISSING_INVENTORY),
        )
        .when(
            F.col("valid_for_mdo_input") != 1,
            F.lit(INVALID_MDO_INPUT),
        )
        .when(
            F.col("valid_for_price_elasticity") != 1,
            F.lit(INVALID_PRICE_ELASTICITY_OUTPUT),
        )
        .when(
            F.col("pe_quantity") < 0,
            F.lit(NEGATIVE_QUANTITY),
        )
        .when(
            F.col("scenario_discount").isNull(),
            F.lit(INVALID_SCENARIO_DISCOUNT),
        )
        .when(
            F.col("scenario_unit_price").isNull()
            | (F.col("scenario_unit_price") <= 0),
            F.lit(INVALID_SCENARIO_PRICE),
        )
        .when(
            F.col("predictive_expected_quantity").isNull(),
            F.lit(MISSING_PREDICTIVE_EXPECTED_QUANTITY),
        )
        .when(
            F.col("predictive_expected_revenue").isNull(),
            F.lit(MISSING_PREDICTIVE_EXPECTED_REVENUE),
        )

        # The current discount is always allowed, even when it is not present
        # in the configured scenario ladder. Only true candidate rows are
        # validated against the configured list.
        .when(
            (F.col("is_candidate_scenario") == 1)
            & (~F.col("scenario_discount_rounded").isin(
                allowed_discount_values
            )),
            F.lit(INVALID_SCENARIO_DISCOUNT),
        )

        # A non-technical row below the current discount is not a valid MDO
        # candidate. The 0% row is retained only for technical PE use.
        .when(
            (F.col("is_technical_baseline") != 1)
            & (F.col("is_current_baseline") != 1)
            & (
                F.col("scenario_discount")
                < F.col("current_discount") - F.lit(0.0005)
            ),
            F.lit(SCENARIO_DISCOUNT_BELOW_CURRENT_DISCOUNT),
        )

        # Maximum discount applies only to new recommendation candidates. The
        # current business position remains available as the baseline even if
        # it is outside the configured ladder.
        .when(
            (F.col("is_candidate_scenario") == 1)
            & (
                F.col("scenario_discount")
                > F.lit(float(max_discount))
            ),
            F.lit(SCENARIO_DISCOUNT_ABOVE_MAX_DISCOUNT),
        )

        # Candidate-specific history and model reliability rules.
        .when(
            (F.col("is_candidate_scenario") == 1)
            & F.col("history_days").isNotNull()
            & (
                F.col("history_days")
                < F.lit(float(min_history_days))
            ),
            F.lit(MINIMUM_HISTORY_NOT_MET),
        )
        .when(
            (F.col("is_candidate_scenario") == 1)
            & F.col("price_variation").isNotNull()
            & (
                F.col("price_variation")
                < F.lit(float(min_price_variation))
            ),
            F.lit(LOW_PRICE_VARIATION),
        )
        .when(
            (F.col("is_candidate_scenario") == 1)
            & F.col("discount_variation").isNotNull()
            & (
                F.col("discount_variation")
                < F.lit(float(min_discount_variation))
            ),
            F.lit(LOW_DISCOUNT_VARIATION),
        )
        .when(
            (F.col("is_candidate_scenario") == 1)
            & F.col("prediction_confidence").isNotNull()
            & (
                F.col("prediction_confidence")
                < F.lit(float(min_prediction_confidence))
            ),
            F.lit(LOW_CONFIDENCE_PREDICTION),
        )

        # Candidate elasticity checks.
        .when(
            (F.col("is_candidate_scenario") == 1)
            & F.col("expected_quantity_elasticity").isNull(),
            F.lit(MISSING_ELASTICITY),
        )
        .when(
            (F.col("is_candidate_scenario") == 1)
            & F.col("expected_quantity_elasticity").isNotNull()
            & (F.col("expected_quantity_elasticity") >= 0),
            F.lit(NEGATIVE_ELASTICITY),
        )
        .when(
            (F.col("is_candidate_scenario") == 1)
            & F.col("expected_quantity_elasticity").isNotNull()
            & (
                F.abs(F.col("expected_quantity_elasticity"))
                > F.lit(float(max_abs_elasticity))
            ),
            F.lit(ELASTICITY_OUTLIER),
        )

        # Candidate comparison requires the current baseline to exist.
        .when(
            (F.col("is_candidate_scenario") == 1)
            & (
                F.col("current_baseline_expected_quantity").isNull()
                | F.col("current_baseline_expected_revenue").isNull()
            ),
            F.lit(INVALID_SCENARIO),
        )

        # Candidate uplift rules now use lifts calculated against the current
        # scenario, not the technical 0% PE baseline.
        .when(
            (F.col("is_candidate_scenario") == 1)
            & F.col("mdo_expected_quantity_lift_pct").isNotNull()
            & (
                F.col("mdo_expected_quantity_lift_pct")
                <= F.lit(float(min_sales_uplift_pct))
            ),
            F.lit(LOW_SALES_UPLIFT),
        )
        .when(
            (F.col("is_candidate_scenario") == 1)
            & (
                F.col("mdo_expected_quantity_lift_pct").isNull()
                | (F.col("mdo_expected_quantity_lift_pct") <= 0)
            ),
            F.lit(INVALID_SCENARIO),
        )
        .when(
            (F.col("is_candidate_scenario") == 1)
            & (
                F.col("mdo_expected_revenue_lift_pct").isNull()
                | (F.col("mdo_expected_revenue_lift_pct") <= 0)
            ),
            F.lit(INVALID_SCENARIO),
        )
    )

    df = (
        df
        .withColumn("reason_code", reason_expr)
        .withColumn(
            "scenario_valid_flag",
            F.when(
                F.col("reason_code").isNull(),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
    )

    return add_reason_priority(df)


# ============================================================
# 4. Build baseline comparison
# ============================================================

def build_baseline_df(scenario_rule_df: DataFrame) -> DataFrame:
    """
    Build the MDO business baseline from the current-discount scenario.

    The 0% scenario remains available in the scenario table for PE technical
    comparison, but it is not used as the MDO baseline when current discount is
    greater than zero.
    """

    baseline_window = Window.partitionBy(*GRAIN_COLUMNS).orderBy(
        F.col("scenario_valid_flag").desc(),
        F.col("capped_expected_revenue").desc_nulls_last(),
    )

    return (
        scenario_rule_df
        .filter(F.col("is_current_baseline") == 1)
        .withColumn(
            "baseline_rank",
            F.row_number().over(baseline_window),
        )
        .filter(F.col("baseline_rank") == 1)
        .select(
            *GRAIN_COLUMNS,
            F.col("scenario_discount").alias("baseline_discount"),
            F.col("capped_expected_quantity").alias(
                "baseline_expected_quantity"
            ),
            F.col("capped_expected_revenue").alias(
                "baseline_expected_revenue"
            ),
            F.col("scenario_unit_price").alias("baseline_unit_price"),
        )
    )


# ============================================================
# 5. Select best valid scenario
# ============================================================

def select_best_revenue_scenario(scenario_rule_df: DataFrame) -> DataFrame:
    """
    Select between the valid current baseline and valid higher-discount
    candidates. A technical 0% row is excluded from selection whenever the
    current discount is greater than zero.
    """

    baseline_df = build_baseline_df(scenario_rule_df)

    valid_df = (
        scenario_rule_df
        .filter(F.col("scenario_valid_flag") == 1)
        .filter(
            (F.col("is_current_baseline") == 1)
            | (F.col("is_candidate_scenario") == 1)
        )
        .join(
            baseline_df,
            on=GRAIN_COLUMNS,
            how="inner",
        )
        .withColumn(
            "calculated_quantity_lift_pct",
            F.when(
                F.col("baseline_expected_quantity") > 0,
                (
                    F.col("capped_expected_quantity")
                    - F.col("baseline_expected_quantity")
                )
                / F.col("baseline_expected_quantity"),
            ).otherwise(
                F.when(
                    F.col("is_current_baseline") == 1,
                    F.lit(0.0),
                )
            ),
        )
        .withColumn(
            "calculated_revenue_lift_pct",
            F.when(
                F.col("baseline_expected_revenue") > 0,
                (
                    F.col("capped_expected_revenue")
                    - F.col("baseline_expected_revenue")
                )
                / F.col("baseline_expected_revenue"),
            ).otherwise(
                F.when(
                    F.col("is_current_baseline") == 1,
                    F.lit(0.0),
                )
            ),
        )
    )

    best_window = Window.partitionBy(*GRAIN_COLUMNS).orderBy(
        F.col("capped_expected_revenue").desc_nulls_last(),
        F.col("scenario_discount").asc_nulls_last(),
    )

    return (
        valid_df
        .withColumn(
            "best_scenario_rank",
            F.row_number().over(best_window),
        )
        .filter(F.col("best_scenario_rank") == 1)
    )


# ============================================================
# 6. Build exclude rows where no valid scenario exists
# ============================================================

def build_no_valid_scenario_exclude_rows(
    scenario_rule_df: DataFrame,
) -> DataFrame:
    """
    Create Exclude rows when neither a valid current baseline nor a valid
    higher-discount candidate is available. A valid technical 0% PE row alone
    must not prevent an Exclude result.
    """

    valid_grain_df = (
        scenario_rule_df
        .filter(F.col("scenario_valid_flag") == 1)
        .filter(
            (F.col("is_current_baseline") == 1)
            | (F.col("is_candidate_scenario") == 1)
        )
        .select(*GRAIN_COLUMNS)
        .distinct()
    )

    invalid_only_df = scenario_rule_df.join(
        valid_grain_df,
        on=GRAIN_COLUMNS,
        how="left_anti",
    )

    # Prefer the current baseline reason, then candidate reasons, and ignore
    # the technical baseline unless it is the only available diagnostic row.
    invalid_window = Window.partitionBy(*GRAIN_COLUMNS).orderBy(
        F.when(F.col("is_current_baseline") == 1, F.lit(0))
        .when(F.col("is_candidate_scenario") == 1, F.lit(1))
        .otherwise(F.lit(2)),
        F.col("reason_priority").asc(),
        F.col("scenario_discount").asc_nulls_last(),
    )

    return (
        invalid_only_df
        .withColumn(
            "invalid_rank",
            F.row_number().over(invalid_window),
        )
        .filter(F.col("invalid_rank") == 1)
        .withColumn(
            "recommended_discount",
            F.lit(None).cast("double"),
        )
        .withColumn(
            "recommended_price",
            F.lit(None).cast("double"),
        )
        .withColumn(
            "recommended_expected_quantity",
            F.lit(None).cast("double"),
        )
        .withColumn(
            "recommended_expected_revenue",
            F.lit(None).cast("double"),
        )
        .withColumn(
            "recommended_quantity_lift_pct",
            F.lit(None).cast("double"),
        )
        .withColumn(
            "recommended_revenue_lift_pct",
            F.lit(None).cast("double"),
        )
        .withColumn(
            "recommended_elasticity",
            F.lit(None).cast("double"),
        )
        .withColumn("action_flag", F.lit(ACTION_EXCLUDE))
        .withColumn(
            "final_reason_code",
            F.coalesce(
                F.col("reason_code"),
                F.lit(NO_VALID_SCENARIO_FOUND),
            ),
        )
    )


# ============================================================
# 7. Build final recommendation
# ============================================================

def build_final_mdo_recommendation(
    scenario_rule_df: DataFrame,
) -> DataFrame:
    """
    Build one final recommendation per article-store-date.

    - Include: a higher-discount candidate has the best positive revenue.
    - No Change: the current-discount scenario is best.
    - Exclude: no valid current/candidate scenario exists.
    """

    best_df = select_best_revenue_scenario(scenario_rule_df)

    best_df = (
        best_df
        .withColumn(
            "_selected_discount",
            F.col("scenario_discount"),
        )
        .withColumn(
            "_selected_price",
            F.col("scenario_unit_price"),
        )
        .withColumn(
            "_selected_quantity",
            F.col("capped_expected_quantity"),
        )
        .withColumn(
            "_selected_revenue",
            F.col("capped_expected_revenue"),
        )
        .withColumn(
            "_selected_quantity_lift_pct",
            F.coalesce(
                F.col("calculated_quantity_lift_pct"),
                F.col("mdo_expected_quantity_lift_pct"),
            ),
        )
        .withColumn(
            "_selected_revenue_lift_pct",
            F.coalesce(
                F.col("calculated_revenue_lift_pct"),
                F.col("mdo_expected_revenue_lift_pct"),
            ),
        )
        .withColumn(
            "action_flag",
            F.when(
                F.col("is_current_baseline") == 1,
                F.lit(ACTION_NO_CHANGE),
            )
            .when(
                F.col("_selected_discount")
                <= F.col("current_discount") + F.lit(0.0005),
                F.lit(ACTION_NO_CHANGE),
            )
            .when(
                F.col("_selected_revenue_lift_pct").isNull()
                | (F.col("_selected_revenue_lift_pct") <= 0),
                F.lit(ACTION_NO_CHANGE),
            )
            .otherwise(F.lit(ACTION_INCLUDE)),
        )

        # For No Change, explicitly return the current discount/current price
        # and current expected results. This prevents 0% from being displayed
        # as No Change when the current discount is greater than zero.
        .withColumn(
            "recommended_discount",
            F.when(
                F.col("action_flag") == ACTION_NO_CHANGE,
                F.col("baseline_discount"),
            ).otherwise(F.col("_selected_discount")),
        )
        .withColumn(
            "recommended_price",
            F.round(
                F.when(
                    F.col("action_flag") == ACTION_NO_CHANGE,
                    F.col("baseline_unit_price"),
                ).otherwise(F.col("_selected_price")),
                4,
            ),
        )
        .withColumn(
            "recommended_expected_quantity",
            F.when(
                F.col("action_flag") == ACTION_NO_CHANGE,
                F.col("baseline_expected_quantity"),
            ).otherwise(F.col("_selected_quantity")),
        )
        .withColumn(
            "recommended_expected_revenue",
            F.when(
                F.col("action_flag") == ACTION_NO_CHANGE,
                F.col("baseline_expected_revenue"),
            ).otherwise(F.col("_selected_revenue")),
        )
        .withColumn(
            "recommended_quantity_lift_pct",
            F.when(
                F.col("action_flag") == ACTION_NO_CHANGE,
                F.lit(0.0),
            ).otherwise(F.col("_selected_quantity_lift_pct")),
        )
        .withColumn(
            "recommended_revenue_lift_pct",
            F.when(
                F.col("action_flag") == ACTION_NO_CHANGE,
                F.lit(0.0),
            ).otherwise(F.col("_selected_revenue_lift_pct")),
        )
        .withColumn(
            "recommended_elasticity",
            F.when(
                F.col("action_flag") == ACTION_NO_CHANGE,
                F.lit(None).cast("double"),
            ).otherwise(F.col("expected_quantity_elasticity")),
        )
        .withColumn(
            "final_reason_code",
            F.when(
                F.col("action_flag") == ACTION_NO_CHANGE,
                F.lit(BASELINE_SCENARIO_IS_BEST),
            )
            .when(
                F.col("action_flag") == ACTION_INCLUDE,
                F.lit(REVENUE_IMPROVES_WITH_MARKDOWN),
            )
            .otherwise(F.lit(NO_REVENUE_IMPROVEMENT)),
        )
    )

    exclude_df = build_no_valid_scenario_exclude_rows(
        scenario_rule_df
    )

    final_columns = [
        "pe_article",
        "pe_store_group",
        "date",
        "current_discount",
        "recommended_discount",
        "base_unit_price",
        "recommended_price",
        "recommended_expected_quantity",
        "recommended_expected_revenue",
        "recommended_quantity_lift_pct",
        "recommended_revenue_lift_pct",
        "recommended_elasticity",
        "action_flag",
        "final_reason_code",
    ]

    return (
        best_df.select(*final_columns)
        .unionByName(
            exclude_df.select(*final_columns),
            allowMissingColumns=True,
        )
        .withColumnRenamed("final_reason_code", "reason_code")
        .withColumn("created_timestamp", F.current_timestamp())
    )


# ============================================================
# 8. End-to-end Phase 1 + Phase 2 MDO rule execution
# ============================================================

def run_mdo_phase1_rules(
    source_df: DataFrame,
    allowed_scenario_discounts: list,
    max_discount: float,
    max_abs_elasticity: float,
    min_history_days: float,
    min_price_variation: float,
    min_discount_variation: float,
    min_prediction_confidence: float,
    min_sales_uplift_pct: float,
):
    """
    Main function used by generic_mdo_phase1_engine.py.

    Returns:
    1. scenario_rule_df
    2. final_recommendation_df
    """

    scenario_rule_df = apply_mdo_phase1_scenario_rules(
        source_df=source_df,
        allowed_scenario_discounts=allowed_scenario_discounts,
        max_discount=max_discount,
        max_abs_elasticity=max_abs_elasticity,
        min_history_days=min_history_days,
        min_price_variation=min_price_variation,
        min_discount_variation=min_discount_variation,
        min_prediction_confidence=min_prediction_confidence,
        min_sales_uplift_pct=min_sales_uplift_pct,
    )

    final_recommendation_df = build_final_mdo_recommendation(
        scenario_rule_df=scenario_rule_df
    )

    return scenario_rule_df, final_recommendation_df