"""
MDO Phase 1 and Phase 2 reason codes.

Purpose:
- Keep all MDO reason codes in one place.
- Keep Include / Exclude / No Change action flags in one place.
- Maintain reason priority so the most important failure reason is shown first.
"""


# ============================================================
# 1. Exclude reason codes
# ============================================================

# Rule 41 - Missing Price Reason Rule:
# Use this reason when price is missing, zero, or invalid because markdown price cannot be calculated.
MISSING_PRICE = "Missing Price"


# Rule 42 - Missing Stock Reason Rule:
# Use this reason when stock is missing or not available because markdown is useful only when stock exists.
MISSING_STOCK = "Missing Stock"


# Rule 43 - Missing Inventory Reason Rule:
# Use this reason when inventory data is missing because MDO needs on-hand inventory for safe recommendations.
MISSING_INVENTORY = "Missing Inventory"


# Rule 1 - MDO Input Eligibility Rule:
# Use this reason when the row is not eligible for MDO input because price, stock, or inventory checks failed.
INVALID_MDO_INPUT = "Invalid MDO Input"


# Rule 29 - Price Elasticity Confidence Rule:
# Use this reason when PE output is not valid because MDO should trust only valid elasticity results.
INVALID_PRICE_ELASTICITY_OUTPUT = "Invalid Price Elasticity Output"


# Rule 5 - Negative Quantity Rule:
# Use this reason when sales quantity is negative because sales quantity cannot be negative for MDO.
NEGATIVE_QUANTITY = "Negative Quantity"


# Rule 30 - Minimum History Rule:
# Use this reason when article-store does not have enough historical sales/price records for reliable MDO recommendation.
MINIMUM_HISTORY_NOT_MET = "Minimum History Not Met"


# Rule 31 - Price Variation Rule:
# Use this reason when price variation is too low, because elasticity cannot be trusted without enough price movement.
LOW_PRICE_VARIATION = "Low Price Variation"


# Rule 32 - Discount Variation Rule:
# Use this reason when discount variation is too low, because MDO needs enough discount movement to evaluate markdown impact.
LOW_DISCOUNT_VARIATION = "Low Discount Variation"


# Rule 15 - Negative Elasticity Rule:
# Use this reason when elasticity is not negative for markdown scenarios, because price decrease should generally increase quantity.
NEGATIVE_ELASTICITY = "Negative Elasticity"


# Rule 33 - Low Confidence Prediction Rule:
# Use this reason when model prediction confidence is below threshold and recommendation is not reliable.
LOW_CONFIDENCE_PREDICTION = "Low Confidence Prediction"


# Rule 35 - Sales Uplift Safety Rule:
# Use this reason when recommended markdown does not produce positive expected sales uplift.
LOW_SALES_UPLIFT = "Low Sales Uplift"


# Rule 7 - Scenario Discount Rule:
# Use this reason when scenario discount is not one of the allowed PE-generated discount levels.
INVALID_SCENARIO_DISCOUNT = "Invalid Scenario Discount"


# Rule 8 - Scenario Price Rule:
# Use this reason when scenario selling price is missing, zero, or negative.
INVALID_SCENARIO_PRICE = "Invalid Scenario Price"


# Rule 9 - Current Discount Protection Rule:
# Use this reason when scenario discount is lower than current discount.
SCENARIO_DISCOUNT_BELOW_CURRENT_DISCOUNT = (
    "Scenario Discount Lower Than Current Discount"
)


# Rule 10 - Maximum Discount Rule:
# Use this reason when scenario discount is higher than the configured maximum discount.
SCENARIO_DISCOUNT_ABOVE_MAX_DISCOUNT = (
    "Scenario Discount Above Max Discount"
)


# Rule 17 - Prediction Availability Rule:
# Use this reason when expected quantity prediction is missing.
MISSING_PREDICTIVE_EXPECTED_QUANTITY = (
    "Missing Predictive Expected Quantity"
)


# Rule 18 - Expected Revenue Availability Rule:
# Use this reason when expected revenue prediction is missing.
MISSING_PREDICTIVE_EXPECTED_REVENUE = (
    "Missing Predictive Expected Revenue"
)


# Rule 14 - Elasticity Validity Rule:
# Use this reason when elasticity is missing for markdown scenarios.
MISSING_ELASTICITY = "Missing Elasticity"


# Rule 16 - Elasticity Outlier Rule:
# Use this reason when elasticity is abnormal or too high to trust.
ELASTICITY_OUTLIER = "Elasticity Outlier"


# Rule 44 - Invalid Scenario Rule:
# Use this reason when the scenario is incomplete or does not pass lift/revenue checks.
INVALID_SCENARIO = "Invalid Scenario"


# ============================================================
# 2. Final action reason codes
# ============================================================

# Rule 24 - No Change Rule:
# Use this reason when 0% discount or current discount is already the best option.
BASELINE_SCENARIO_IS_BEST = "Baseline Scenario Is Best"


# Rule 25 - Include Rule:
# Use this reason when markdown improves expected revenue and should be recommended.
REVENUE_IMPROVES_WITH_MARKDOWN = "Revenue Improves With Markdown"


# Rule 24 - No Change Rule:
# Use this reason when markdown does not improve expected revenue.
NO_REVENUE_IMPROVEMENT = "No Revenue Improvement"


# Rule 26 - Exclude Rule:
# Use this reason when no valid discount scenario is available for the article-store-date.
NO_VALID_SCENARIO_FOUND = "No Valid Scenario Found"


# Rule 50 - MDO Readiness Rule:
# Use this reason when data readiness is below the configured threshold.
MDO_READINESS_BELOW_THRESHOLD = "MDO Readiness Below Threshold"


# ============================================================
# 3. Action flags
# ============================================================

# Rule 25 - Include Rule:
# Final action when MDO recommends applying a higher markdown discount.
ACTION_INCLUDE = "Include"


# Rule 26 - Exclude Rule:
# Final action when MDO cannot safely recommend because data or scenario checks failed.
ACTION_EXCLUDE = "Exclude"


# Rule 24 - No Change Rule:
# Final action when no markdown change is required.
ACTION_NO_CHANGE = "No Change"


# ============================================================
# 4. Reason priority
# Lower number = higher priority
# ============================================================

# Rule 28 - Reason Priority Rule:
# If multiple rules fail, show the highest-priority reason so business users see the main issue first.
REASON_PRIORITY = {
    MISSING_PRICE: 1,
    MISSING_STOCK: 2,
    MISSING_INVENTORY: 3,
    INVALID_MDO_INPUT: 4,
    INVALID_PRICE_ELASTICITY_OUTPUT: 5,
    NEGATIVE_QUANTITY: 6,

    MINIMUM_HISTORY_NOT_MET: 7,
    LOW_PRICE_VARIATION: 8,
    LOW_DISCOUNT_VARIATION: 9,
    NEGATIVE_ELASTICITY: 10,
    LOW_CONFIDENCE_PREDICTION: 11,
    LOW_SALES_UPLIFT: 12,

    INVALID_SCENARIO_DISCOUNT: 13,
    INVALID_SCENARIO_PRICE: 14,
    SCENARIO_DISCOUNT_BELOW_CURRENT_DISCOUNT: 15,
    SCENARIO_DISCOUNT_ABOVE_MAX_DISCOUNT: 16,
    MISSING_PREDICTIVE_EXPECTED_QUANTITY: 17,
    MISSING_PREDICTIVE_EXPECTED_REVENUE: 18,
    MISSING_ELASTICITY: 19,
    ELASTICITY_OUTLIER: 20,
    INVALID_SCENARIO: 21,
    NO_VALID_SCENARIO_FOUND: 22,
    MDO_READINESS_BELOW_THRESHOLD: 23,
    NO_REVENUE_IMPROVEMENT: 24,
    BASELINE_SCENARIO_IS_BEST: 25,
    REVENUE_IMPROVES_WITH_MARKDOWN: 26,
}


# ============================================================
# 5. Helper functions
# ============================================================

def get_reason_priority(reason_code: str) -> int:
    """
    Returns priority for a reason code.
    Unknown reasons get low priority.
    """
    return REASON_PRIORITY.get(reason_code, 999)


def get_all_reason_codes() -> list:
    """
    Returns all supported reason codes.
    """
    return list(REASON_PRIORITY.keys())


def is_valid_reason_code(reason_code: str) -> bool:
    """
    Checks whether reason code is configured.
    """
    return reason_code in REASON_PRIORITY