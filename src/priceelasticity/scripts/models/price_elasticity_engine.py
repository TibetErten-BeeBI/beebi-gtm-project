from pyspark.sql import SparkSession


# ============================================================
# 1. Spark setup
# ============================================================

spark = SparkSession.builder.getOrCreate()


# ============================================================
# 2. Table config
# ============================================================

SOURCE_SCENARIO_TABLE = "workspace.default.pe_predictive_scenario_output"

OUTPUT_PE_ENGINE_TABLE = "workspace.default.pe_price_elasticity_engine_output_dev"


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
    print("Running final PE price elasticity engine output")
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
                pe_unit_price AS base_unit_price,
                predictive_expected_quantity * pe_unit_price AS base_expected_revenue
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

                s.pe_unit_price AS base_unit_price,
                s.pe_unit_price * (1 - s.scenario_discount) AS scenario_unit_price,

                s.actual_quantity,

                s.predicted_deuplifted_quantity,
                s.predictive_scenario_quantity,
                s.predictive_scenario_probability,
                s.predictive_expected_quantity,

                s.sales_scenario_discount_effect_raw,
                s.sales_scenario_discount_effect,

                s.use_sales_arma_correction,
                s.arma_residual_correction,

                s.valid_for_price_elasticity,
                s.valid_for_mdo_input,

                b.base_scenario_quantity,
                b.base_scenario_probability,
                b.base_expected_quantity,
                b.base_expected_revenue,

                s.predictive_expected_quantity * s.pe_unit_price * (1 - s.scenario_discount) AS scenario_expected_revenue,

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
                        (s.predictive_expected_quantity * s.pe_unit_price * (1 - s.scenario_discount))
                        - b.base_expected_revenue
                    ) / b.base_expected_revenue
                    ELSE NULL
                END AS expected_revenue_lift_pct,

                CASE
                    WHEN s.scenario_discount > 0
                    THEN -1 * s.scenario_discount
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
                AVG(scenario_quantity_elasticity) AS avg_scenario_quantity_elasticity
            FROM {OUTPUT_PE_ENGINE_TABLE}
            GROUP BY scenario_discount
            ORDER BY scenario_discount
        """)
    )

    print("Validation 3: sample final output")
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
                expected_quantity_lift_pct,
                scenario_quantity_lift_pct,
                expected_revenue_lift_pct,
                expected_quantity_elasticity,
                scenario_quantity_elasticity,
                use_sales_arma_correction,
                valid_for_price_elasticity,
                valid_for_mdo_input
            FROM {OUTPUT_PE_ENGINE_TABLE}
            ORDER BY pe_article, pe_store_group, wm_yr_wk, scenario_discount
            LIMIT 50
        """)
    )

    print("Final PE price elasticity engine completed successfully.")

    return OUTPUT_PE_ENGINE_TABLE


# ============================================================
# 5. Runner
# ============================================================

if __name__ == "__main__":
    run_final_price_elasticity_output()