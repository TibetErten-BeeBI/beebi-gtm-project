import sys
from pyspark.sql import SparkSession


# ============================================================
# 1. Spark setup
# ============================================================

spark = SparkSession.builder.getOrCreate()


# ============================================================
# 2. Table config - generic table naming
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


OUTPUT_SCHEMA = get_param("output_schema", "workspace.default")
TABLE_PREFIX = get_param("table_prefix", "")


def table_name(base_name: str) -> str:
    if TABLE_PREFIX:
        return f"{OUTPUT_SCHEMA}.{TABLE_PREFIX}_{base_name}"
    return f"{OUTPUT_SCHEMA}.{base_name}"


SOURCE_SCENARIO_TABLE = table_name("pe_predictive_scenario_output")

OUTPUT_PE_ENGINE_TABLE = table_name("pe_price_elasticity_engine_output_dev")
PE_MODEL_VALIDATION_TABLE = table_name("pe_model_validation_summary")
MDO_INPUT_READY_TABLE = table_name("mdo_input_ready")


# ============================================================
# 3. Helper
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


# ============================================================
# 4. Final PE engine output
# ============================================================

def run_final_price_elasticity_output():
    print("==================================================")
    print("Running final PE price elasticity engine output with Phase 2 support columns")
    print("==================================================")

    require_table(SOURCE_SCENARIO_TABLE)

    spark.sql(f"""
        CREATE OR REPLACE TABLE {OUTPUT_PE_ENGINE_TABLE} AS
        WITH scenario_base AS (
            SELECT
                date,
                wm_yr_wk,
                pe_article,
                pe_store_group,
                pe_article_store_group,

                predictive_scenario_quantity AS base_scenario_quantity,
                predictive_scenario_probability AS base_scenario_probability,
                predictive_expected_quantity AS base_expected_quantity,

                pe_unit_price AS current_unit_price,
                scenario_unit_price AS base_unit_price,

                predictive_expected_quantity * scenario_unit_price AS base_expected_revenue
            FROM {SOURCE_SCENARIO_TABLE}
            WHERE scenario_discount = 0
        ),

        scenario_all AS (
            SELECT
                s.date,
                s.wm_yr_wk,
                s.pe_article,
                s.pe_store_group,
                s.pe_article_store_group,

                s.pe_category,
                s.pe_product_type,
                s.pe_product_division,
                s.pe_country,

                s.current_discount,
                s.scenario_discount,

                b.base_unit_price AS base_unit_price,
                s.scenario_unit_price AS scenario_unit_price,

                s.actual_quantity,

                s.predicted_deuplifted_quantity,
                s.predictive_scenario_quantity,
                s.predictive_scenario_probability,
                s.predictive_expected_quantity,

                s.sales_scenario_discount_effect_raw,
                s.sales_scenario_discount_effect,

                s.discount_effect_coefficient_1,
                s.discount_effect_coefficient_2,
                s.discount_effect_coefficient_3,
                s.discount_effect_coefficient_4,
                s.coefficient_level,
                s.coefficient_source,
                s.estimated_elasticity,
                s.elasticity_source,
                s.elasticity_status,
                s.scenario_price_ratio,
                s.coefficient_model_status,
                s.random_effect_applied,
                s.coefficient_history_rows,
                s.coefficient_discount_variation_count,

                -- Rule 30 - Minimum History Rule:
                -- Carry history_days forward so MDO can exclude rows with insufficient history.
                s.history_days,

                -- Rule 31 - Price Variation Rule:
                -- Carry price_variation forward so MDO can exclude rows with low price movement.
                s.price_variation,

                -- Rule 32 - Discount Variation Rule:
                -- Carry discount_variation forward so MDO can exclude rows with low markdown movement.
                s.discount_variation,

                -- Rule 33 - Low Confidence Prediction Rule:
                -- Carry prediction_confidence forward when available so MDO can exclude low-confidence scenarios.
                s.prediction_confidence,

                s.use_sales_arma_correction,
                s.use_sales_residual_correction,
                s.arma_residual_correction,

                s.valid_for_price_elasticity,
                s.valid_for_mdo_input,

                s.pe_store_stock_quantity,
                s.pe_inventory_onhand_quantity,
                s.stock_available_flag,
                s.inventory_available_flag,

                b.base_scenario_quantity,
                b.base_scenario_probability,
                b.base_expected_quantity,
                b.base_expected_revenue,

                s.predictive_expected_quantity * s.scenario_unit_price AS scenario_expected_revenue,

                CASE
                    WHEN b.base_expected_quantity > 0
                    THEN (s.predictive_expected_quantity - b.base_expected_quantity) / b.base_expected_quantity
                    ELSE NULL
                END AS expected_quantity_lift_pct,

                CASE
                    WHEN b.base_scenario_quantity > 0
                    THEN (s.predictive_scenario_quantity - b.base_scenario_quantity) / b.base_scenario_quantity
                    ELSE NULL
                END AS scenario_quantity_lift_pct,

                CASE
                    WHEN b.base_expected_revenue > 0
                    THEN (
                        (s.predictive_expected_quantity * s.scenario_unit_price)
                        - b.base_expected_revenue
                    ) / b.base_expected_revenue
                    ELSE NULL
                END AS expected_revenue_lift_pct,

                CASE
                    WHEN s.scenario_discount > 0
                         AND b.base_unit_price > 0
                         AND s.scenario_unit_price IS NOT NULL
                    THEN (s.scenario_unit_price - b.base_unit_price) / b.base_unit_price
                    ELSE NULL
                END AS price_change_pct,

                s.model_created_at
            FROM {SOURCE_SCENARIO_TABLE} s
            INNER JOIN scenario_base b
                ON s.date = b.date
                AND s.wm_yr_wk = b.wm_yr_wk
                AND s.pe_article = b.pe_article
                AND s.pe_store_group = b.pe_store_group
                AND s.pe_article_store_group = b.pe_article_store_group
        )

        SELECT
            *,

            CASE
                WHEN price_change_pct IS NOT NULL
                     AND ABS(price_change_pct) > 0
                     AND expected_quantity_lift_pct IS NOT NULL
                THEN expected_quantity_lift_pct / price_change_pct
                ELSE NULL
            END AS expected_quantity_elasticity,

            CASE
                WHEN price_change_pct IS NOT NULL
                     AND ABS(price_change_pct) > 0
                     AND scenario_quantity_lift_pct IS NOT NULL
                THEN scenario_quantity_lift_pct / price_change_pct
                ELSE NULL
            END AS scenario_quantity_elasticity

        FROM scenario_all
    """)

    print(f"Final PE engine output table created: {OUTPUT_PE_ENGINE_TABLE}")

    # ========================================================
    # PE model validation summary
    # ========================================================

    spark.sql(f"""
        CREATE OR REPLACE TABLE {PE_MODEL_VALIDATION_TABLE} AS
        SELECT
            COUNT(*) AS final_output_rows,

            SUM(CASE WHEN base_expected_quantity IS NULL THEN 1 ELSE 0 END) AS null_base_qty,
            SUM(CASE WHEN predictive_expected_quantity IS NULL THEN 1 ELSE 0 END) AS null_scenario_qty,
            SUM(CASE WHEN scenario_expected_revenue IS NULL THEN 1 ELSE 0 END) AS null_revenue,

            SUM(CASE WHEN predictive_expected_quantity < 0 THEN 1 ELSE 0 END) AS negative_scenario_qty,
            SUM(CASE WHEN scenario_expected_revenue < 0 THEN 1 ELSE 0 END) AS negative_revenue,

            COUNT(DISTINCT scenario_discount) AS scenario_discount_count,
            COUNT(DISTINCT pe_article) AS product_count,
            COUNT(DISTINCT pe_store_group) AS store_count,
            COUNT(DISTINCT pe_article_store_group) AS product_store_count,

            AVG(CASE WHEN scenario_discount > 0 THEN expected_quantity_elasticity END) AS avg_expected_quantity_elasticity,
            MIN(CASE WHEN scenario_discount > 0 THEN expected_quantity_elasticity END) AS min_expected_quantity_elasticity,
            MAX(CASE WHEN scenario_discount > 0 THEN expected_quantity_elasticity END) AS max_expected_quantity_elasticity,
            STDDEV(CASE WHEN scenario_discount > 0 THEN expected_quantity_elasticity END) AS expected_quantity_elasticity_stddev,

            AVG(CASE WHEN scenario_discount > 0 THEN scenario_quantity_elasticity END) AS avg_scenario_quantity_elasticity,
            MIN(CASE WHEN scenario_discount > 0 THEN scenario_quantity_elasticity END) AS min_scenario_quantity_elasticity,
            MAX(CASE WHEN scenario_discount > 0 THEN scenario_quantity_elasticity END) AS max_scenario_quantity_elasticity,
            STDDEV(CASE WHEN scenario_discount > 0 THEN scenario_quantity_elasticity END) AS scenario_quantity_elasticity_stddev,

            SUM(CASE WHEN scenario_discount > 0 AND expected_quantity_elasticity > 0 THEN 1 ELSE 0 END) AS positive_expected_elasticity_rows,
            SUM(CASE WHEN scenario_discount > 0 AND expected_quantity_elasticity < 0 THEN 1 ELSE 0 END) AS negative_expected_elasticity_rows,

            AVG(discount_effect_coefficient_1) AS avg_discount_effect_coefficient_1,
            MIN(discount_effect_coefficient_1) AS min_discount_effect_coefficient_1,
            MAX(discount_effect_coefficient_1) AS max_discount_effect_coefficient_1,
            STDDEV(discount_effect_coefficient_1) AS discount_effect_coefficient_1_stddev,

            AVG(estimated_elasticity) AS avg_estimated_elasticity,
            MIN(estimated_elasticity) AS min_estimated_elasticity,
            MAX(estimated_elasticity) AS max_estimated_elasticity,
            STDDEV(estimated_elasticity) AS estimated_elasticity_stddev,

            -- Rule 30 - Minimum History Rule:
            -- Summarize history_days to confirm Phase 2 history rule input is available.
            AVG(history_days) AS avg_history_days,
            MIN(history_days) AS min_history_days,
            MAX(history_days) AS max_history_days,

            -- Rule 31 - Price Variation Rule:
            -- Summarize price variation to confirm Phase 2 price variation rule input is available.
            AVG(price_variation) AS avg_price_variation,
            MIN(price_variation) AS min_price_variation,
            MAX(price_variation) AS max_price_variation,

            -- Rule 32 - Discount Variation Rule:
            -- Summarize discount variation to confirm Phase 2 discount variation rule input is available.
            AVG(discount_variation) AS avg_discount_variation,
            MIN(discount_variation) AS min_discount_variation,
            MAX(discount_variation) AS max_discount_variation,

            -- Rule 33 - Low Confidence Prediction Rule:
            -- Summarize prediction confidence when model output provides it.
            AVG(prediction_confidence) AS avg_prediction_confidence,
            MIN(prediction_confidence) AS min_prediction_confidence,
            MAX(prediction_confidence) AS max_prediction_confidence,

            SUM(CASE WHEN valid_for_mdo_input = 1 THEN 1 ELSE 0 END) AS mdo_ready_rows,

            SUM(
                CASE
                    WHEN valid_for_mdo_input = 1
                         AND scenario_unit_price > 0
                         AND predictive_expected_quantity >= 0
                         AND scenario_expected_revenue >= 0
                         AND pe_store_stock_quantity IS NOT NULL
                         AND pe_inventory_onhand_quantity IS NOT NULL
                    THEN 1 ELSE 0
                END
            ) AS clean_mdo_input_rows,

            CASE
                WHEN SUM(CASE WHEN predictive_expected_quantity < 0 THEN 1 ELSE 0 END) > 0
                THEN 'FAIL_NEGATIVE_QUANTITY'

                WHEN SUM(CASE WHEN scenario_expected_revenue < 0 THEN 1 ELSE 0 END) > 0
                THEN 'FAIL_NEGATIVE_REVENUE'

                WHEN COUNT(DISTINCT scenario_discount) < 2
                THEN 'FAIL_SCENARIO_DISCOUNTS_MISSING'

                WHEN STDDEV(CASE WHEN scenario_discount > 0 THEN expected_quantity_elasticity END) < 0.0001
                THEN 'WARNING_SAME_ELASTICITY_FOR_ALL_PRODUCTS'

                WHEN STDDEV(discount_effect_coefficient_1) < 0.0001
                THEN 'WARNING_SAME_DISCOUNT_COEFFICIENT_FOR_ALL_PRODUCTS'

                ELSE 'PASS'
            END AS validation_status

        FROM {OUTPUT_PE_ENGINE_TABLE}
    """)

    print(f"PE model validation summary created: {PE_MODEL_VALIDATION_TABLE}")

    # ========================================================
    # MDO input-ready table
    # This is not MDO optimization.
    # It only prepares clean rows for future MDO.
    # ========================================================

    spark.sql(f"""
        CREATE OR REPLACE TABLE {MDO_INPUT_READY_TABLE} AS
        SELECT *
        FROM {OUTPUT_PE_ENGINE_TABLE}
        WHERE valid_for_mdo_input = 1
          AND scenario_unit_price > 0
          AND predictive_expected_quantity >= 0
          AND scenario_expected_revenue >= 0
          AND pe_store_stock_quantity IS NOT NULL
          AND pe_inventory_onhand_quantity IS NOT NULL
    """)

    print(f"MDO input-ready table created: {MDO_INPUT_READY_TABLE}")

    # ========================================================
    # Notebook / job validations
    # ========================================================

    print("Validation 1: final output row count")
    display(
        spark.sql(f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT pe_article) AS products,
                COUNT(DISTINCT pe_store_group) AS stores,
                COUNT(DISTINCT pe_article_store_group) AS product_store_groups,
                COUNT(DISTINCT wm_yr_wk) AS weeks,
                MIN(date) AS min_date,
                MAX(date) AS max_date
            FROM {OUTPUT_PE_ENGINE_TABLE}
        """)
    )

    print("Validation 2: scenario discount row counts")
    display(
        spark.sql(f"""
            SELECT
                scenario_discount,
                COUNT(*) AS rows,
                AVG(expected_quantity_lift_pct) AS avg_expected_quantity_lift_pct,
                AVG(scenario_quantity_lift_pct) AS avg_scenario_quantity_lift_pct,
                AVG(expected_revenue_lift_pct) AS avg_expected_revenue_lift_pct,
                AVG(expected_quantity_elasticity) AS avg_expected_quantity_elasticity,
                AVG(scenario_quantity_elasticity) AS avg_scenario_quantity_elasticity,
                AVG(discount_effect_coefficient_1) AS avg_discount_effect_coefficient_1,
                AVG(estimated_elasticity) AS avg_estimated_elasticity,
                MIN(discount_effect_coefficient_1) AS min_discount_effect_coefficient_1,
                MAX(discount_effect_coefficient_1) AS max_discount_effect_coefficient_1,
                STDDEV(discount_effect_coefficient_1) AS stddev_discount_effect_coefficient_1
            FROM {OUTPUT_PE_ENGINE_TABLE}
            GROUP BY scenario_discount
            ORDER BY scenario_discount
        """)
    )

    print("Validation 3: PE model validation summary")
    display(
        spark.sql(f"""
            SELECT *
            FROM {PE_MODEL_VALIDATION_TABLE}
        """)
    )

    print("Validation 4: coefficient-level distribution")
    display(
        spark.sql(f"""
            SELECT
                coefficient_level,
                coefficient_source,
                elasticity_source,
                elasticity_status,
                coefficient_model_status,
                COUNT(*) AS rows,
                AVG(discount_effect_coefficient_1) AS avg_coef_1,
                AVG(estimated_elasticity) AS avg_estimated_elasticity,
                MIN(discount_effect_coefficient_1) AS min_coef_1,
                MAX(discount_effect_coefficient_1) AS max_coef_1,
                STDDEV(discount_effect_coefficient_1) AS stddev_coef_1,
                AVG(expected_quantity_elasticity) AS avg_expected_quantity_elasticity
            FROM {OUTPUT_PE_ENGINE_TABLE}
            WHERE scenario_discount > 0
            GROUP BY
                coefficient_level,
                coefficient_source,
                elasticity_source,
                elasticity_status,
                coefficient_model_status
            ORDER BY coefficient_level, coefficient_source
        """)
    )

    print("Validation 5: product-level elasticity variation")
    display(
        spark.sql(f"""
            SELECT
                pe_article,
                COUNT(DISTINCT pe_article_store_group) AS product_store_groups,
                AVG(expected_quantity_elasticity) AS avg_expected_quantity_elasticity,
                MIN(expected_quantity_elasticity) AS min_expected_quantity_elasticity,
                MAX(expected_quantity_elasticity) AS max_expected_quantity_elasticity,
                STDDEV(expected_quantity_elasticity) AS stddev_expected_quantity_elasticity,
                AVG(discount_effect_coefficient_1) AS avg_discount_effect_coefficient_1,
                AVG(estimated_elasticity) AS avg_estimated_elasticity
            FROM {OUTPUT_PE_ENGINE_TABLE}
            WHERE scenario_discount > 0
            GROUP BY pe_article
            ORDER BY avg_expected_quantity_elasticity
        """)
    )

    print("Validation 6: MDO input-ready row count")
    display(
        spark.sql(f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT pe_article) AS products,
                COUNT(DISTINCT pe_store_group) AS stores,
                COUNT(DISTINCT pe_article_store_group) AS product_store_groups,
                COUNT(DISTINCT scenario_discount) AS scenario_discount_count,
                MIN(date) AS min_date,
                MAX(date) AS max_date
            FROM {MDO_INPUT_READY_TABLE}
        """)
    )

    print("Validation 7: sample final output")
    display(
        spark.sql(f"""
            SELECT
                date,
                wm_yr_wk,
                pe_article,
                pe_store_group,
                scenario_discount,
                base_unit_price,
                scenario_unit_price,
                actual_quantity,
                predictive_scenario_quantity,
                predictive_scenario_probability,
                predictive_expected_quantity,
                base_expected_quantity,
                scenario_expected_revenue,
                expected_quantity_lift_pct,
                scenario_quantity_lift_pct,
                expected_revenue_lift_pct,
                price_change_pct,
                expected_quantity_elasticity,
                scenario_quantity_elasticity,
                scenario_price_ratio,
                estimated_elasticity,
                elasticity_source,
                elasticity_status,
                discount_effect_coefficient_1,
                coefficient_level,
                coefficient_source,
                coefficient_model_status,
                use_sales_arma_correction,
                valid_for_price_elasticity,
                valid_for_mdo_input,
                pe_store_stock_quantity,
                pe_inventory_onhand_quantity,
                history_days,
                price_variation,
                discount_variation,
                prediction_confidence
            FROM {OUTPUT_PE_ENGINE_TABLE}
            ORDER BY pe_article, pe_store_group, date, scenario_discount
            LIMIT 100
        """)
    )

    print("Final PE price elasticity engine completed successfully.")

    return OUTPUT_PE_ENGINE_TABLE


# ============================================================
# 5. Runner
# ============================================================

if __name__ == "__main__":
    run_final_price_elasticity_output()