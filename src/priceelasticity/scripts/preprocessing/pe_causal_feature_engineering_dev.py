import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# 1. Spark setup
# ============================================================

spark = SparkSession.builder.getOrCreate()


# ============================================================
# 2. Config
# ============================================================

SOURCE_BASE_TABLE = "workspace.default.base_data_table"

OUTPUT_FEATURE_TABLE = "workspace.default.pe_causal_features_dev"
OUTPUT_FEATURE_VIEW = "pe_causal_features_dev_view"

OUTPUT_FEATURE_QUALITY_TABLE = "workspace.default.pe_causal_features_quality_summary"

MIN_HISTORY_POINTS_FOR_CAUSAL = 4

# This is used to convert probability 0/1 into logit safely.
# 0 becomes 0.01, 1 becomes 0.99.
PROBABILITY_EPSILON = 0.01


# ============================================================
# 3. Helpers
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


def first_existing_column(df, candidate_columns):
    for col_name in candidate_columns:
        if col_name in df.columns:
            return col_name
    return None


def require_columns(df, required_columns, label):
    missing_columns = [
        col_name
        for col_name in required_columns
        if col_name not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {label}: {missing_columns}"
        )


def safe_select_col(df, source_col, alias_col=None, cast_type=None, default_value=None):
    alias_col = alias_col or source_col

    if source_col in df.columns:
        expr = F.col(source_col)

        if cast_type:
            expr = expr.cast(cast_type)

        return expr.alias(alias_col)

    if default_value is not None:
        expr = F.lit(default_value)
    else:
        expr = F.lit(None)

    if cast_type:
        expr = expr.cast(cast_type)

    return expr.alias(alias_col)


# ============================================================
# 4. Load base table
# ============================================================

def load_base_table():
    print("==============================================")
    print("Loading PE base table")
    print("==============================================")

    require_table(SOURCE_BASE_TABLE)

    base_df = spark.table(SOURCE_BASE_TABLE)

    print("Base table:", SOURCE_BASE_TABLE)
    print("Base rows:", base_df.count())
    print("Base columns:", len(base_df.columns))

    require_columns(
        base_df,
        [
            "date",
            "wm_yr_wk",
            "pe_article",
            "pe_store_group",
            "pe_quantity",
            "is_sold",
            "probability_target",
        ],
        "base_data_table",
    )

    return base_df


# ============================================================
# 5. Prepare standard feature base
# ============================================================

def prepare_standard_feature_base(base_df):
    """
    Convert base_data_table into a clean product-store-week feature table.

    This keeps all rows, including rows where some price/stock/inventory values
    may be missing. Later model steps use quality flags to filter valid rows.
    """

    print("Preparing standard PE feature base...")

    feature_base_df = (
        base_df
        .select(
            F.to_date(F.col("date")).alias("date"),
            safe_select_col(base_df, "week_start_date", "week_start_date"),
            F.col("wm_yr_wk").cast("int").alias("wm_yr_wk"),
            safe_select_col(base_df, "calendar_month", "calendar_month", "int"),
            safe_select_col(base_df, "calendar_year", "calendar_year", "int"),

            F.col("pe_article").alias("pe_article"),
            F.col("pe_store_group").alias("pe_store_group"),
            safe_select_col(base_df, "pe_article_store_group", "pe_article_store_group"),

            F.col("pe_quantity").cast("double").alias("pe_quantity"),
            safe_select_col(base_df, "days_in_week", "days_in_week", "int"),
            safe_select_col(base_df, "days_sold_count", "days_sold_count", "int"),

            safe_select_col(base_df, "pe_unit_price", "pe_unit_price", "double"),
            safe_select_col(base_df, "pe_actual_retail_price", "pe_actual_retail_price", "double"),
            safe_select_col(base_df, "pe_average_zone_retail_price", "pe_average_zone_retail_price", "double"),
            safe_select_col(base_df, "discount", "discount", "double"),
            safe_select_col(base_df, "pe_sales_amount", "pe_sales_amount", "double"),

            safe_select_col(base_df, "pe_store_stock_quantity", "pe_store_stock_quantity", "double"),
            safe_select_col(base_df, "pe_inventory_onhand_quantity", "pe_inventory_onhand_quantity", "double"),
            safe_select_col(base_df, "pe_store_stock_quantity_for_model", "pe_store_stock_quantity_for_model", "double", 0.0),
            safe_select_col(base_df, "pe_inventory_onhand_quantity_for_model", "pe_inventory_onhand_quantity_for_model", "double", 0.0),

            F.col("is_sold").cast("double").alias("is_sold"),
            F.col("probability_target").cast("double").alias("probability_target"),

            safe_select_col(base_df, "valid_price_flag", "valid_price_flag", "int", 0),
            safe_select_col(base_df, "price_available_flag", "price_available_flag", "int", 0),
            safe_select_col(base_df, "stock_available_flag", "stock_available_flag", "int", 0),
            safe_select_col(base_df, "inventory_available_flag", "inventory_available_flag", "int", 0),
            safe_select_col(base_df, "valid_for_price_elasticity", "valid_for_price_elasticity", "int", 0),
            safe_select_col(base_df, "valid_for_mdo_input", "valid_for_mdo_input", "int", 0),

            safe_select_col(base_df, "missing_price_reason", "missing_price_reason"),
            safe_select_col(base_df, "missing_stock_reason", "missing_stock_reason"),
            safe_select_col(base_df, "missing_inventory_reason", "missing_inventory_reason"),

            safe_select_col(base_df, "price_data_quality_flag", "price_data_quality_flag"),
            safe_select_col(base_df, "price_start_date", "price_start_date"),
            safe_select_col(base_df, "price_end_date", "price_end_date"),

            safe_select_col(base_df, "pe_gender", "pe_gender"),
            safe_select_col(base_df, "pe_category", "pe_category"),
            safe_select_col(base_df, "pe_product_type", "pe_product_type"),
            safe_select_col(base_df, "pe_product_division", "pe_product_division"),
            safe_select_col(base_df, "pe_store_state", "pe_store_state"),
            safe_select_col(base_df, "pe_store_type", "pe_store_type"),
            safe_select_col(base_df, "pe_country", "pe_country"),

            safe_select_col(base_df, "event_name_1", "event_name_1"),
            safe_select_col(base_df, "event_type_1", "event_type_1"),
            safe_select_col(base_df, "event_name_2", "event_name_2"),
            safe_select_col(base_df, "event_type_2", "event_type_2"),
            safe_select_col(base_df, "snap", "snap", "int", 0),

            safe_select_col(base_df, "base_data_hash_id", "base_data_hash_id"),
            safe_select_col(base_df, "base_data_grain", "base_data_grain"),
        )
        .filter(F.col("date").isNotNull())
        .filter(F.col("wm_yr_wk").isNotNull())
        .filter(F.col("pe_article").isNotNull())
        .filter(F.col("pe_store_group").isNotNull())
        .filter(F.col("pe_quantity").isNotNull())
        .filter(F.col("pe_quantity") >= 0)
    )

    feature_base_df = feature_base_df.withColumn(
        "pe_article_store_group",
        F.when(
            F.col("pe_article_store_group").isNotNull(),
            F.col("pe_article_store_group")
        ).otherwise(
            F.concat_ws("_", F.col("pe_article"), F.col("pe_store_group"))
        )
    )

    print("Standard feature base rows:", feature_base_df.count())

    return feature_base_df


# ============================================================
# 6. Add core model features
# ============================================================

def add_core_model_features(feature_df):
    """
    Add sales target, price features, stock/inventory features,
    discount powers, and probability logit target.
    """

    print("Adding core model features...")

    feature_df = (
        feature_df
        .withColumn(
            "log_quantity",
            F.when(F.col("pe_quantity") > 0, F.log(F.col("pe_quantity")))
             .otherwise(F.lit(None).cast("double"))
        )
        .withColumn(
            "log_quantity_plus_one",
            F.log(F.col("pe_quantity") + F.lit(1.0))
        )
        .withColumn(
            "log_price",
            F.when(F.col("pe_unit_price") > 0, F.log(F.col("pe_unit_price")))
             .otherwise(F.lit(None).cast("double"))
        )
        .withColumn(
            "log_sales_amount",
            F.when(
                F.col("pe_sales_amount").isNotNull(),
                F.log(F.col("pe_sales_amount") + F.lit(1.0))
            ).otherwise(F.lit(None).cast("double"))
        )
        .withColumn(
            "log_store_stock_quantity",
            F.log(
                F.coalesce(
                    F.col("pe_store_stock_quantity_for_model"),
                    F.lit(0.0)
                ) + F.lit(1.0)
            )
        )
        .withColumn(
            "log_inventory_onhand_quantity",
            F.log(
                F.coalesce(
                    F.col("pe_inventory_onhand_quantity_for_model"),
                    F.lit(0.0)
                ) + F.lit(1.0)
            )
        )
    )

    feature_df = (
        feature_df
        .withColumn("treatment_price", F.col("log_price"))
        .withColumn("outcome_quantity", F.col("log_quantity"))
        .withColumn("sold_qty_log", F.col("log_quantity"))
        .withColumn("probability_target", F.col("is_sold").cast("double"))
    )

    # Important change:
    # Probability causal/mixed-linear model should not train directly on raw 0/1.
    # So we create smoothed probability and logit target.
    feature_df = (
        feature_df
        .withColumn(
            "probability_target_smoothed",
            F.when(F.col("probability_target") >= 1.0, F.lit(1.0 - PROBABILITY_EPSILON))
             .when(F.col("probability_target") <= 0.0, F.lit(PROBABILITY_EPSILON))
             .otherwise(F.col("probability_target"))
        )
        .withColumn(
            "probability_logit_target",
            F.log(
                F.col("probability_target_smoothed") /
                (F.lit(1.0) - F.col("probability_target_smoothed"))
            )
        )
    )

    feature_df = (
        feature_df
        .withColumn(
            "discount_power_1",
            F.when(F.col("discount").isNotNull(), F.col("discount"))
             .otherwise(F.lit(None).cast("double"))
        )
        .withColumn(
            "discount_power_2",
            F.when(F.col("discount").isNotNull(), F.pow(F.col("discount"), 2))
             .otherwise(F.lit(None).cast("double"))
        )
        .withColumn(
            "discount_power_3",
            F.when(F.col("discount").isNotNull(), F.pow(F.col("discount"), 3))
             .otherwise(F.lit(None).cast("double"))
        )
        .withColumn(
            "discount_power_4",
            F.when(F.col("discount").isNotNull(), F.pow(F.col("discount"), 4))
             .otherwise(F.lit(None).cast("double"))
        )
    )

    return feature_df


# ============================================================
# 7. Add CCE features
# ============================================================

def add_cce_features(feature_df):
    """
    Add CCE columns.

    CCE = Common Correlated Effects.

    These columns summarize common demand and price movement by week,
    store-week, category-week, product-type-week, division-week, and country-week.
    The causal model later uses these for orthogonalization.
    """

    print("Adding CCE features...")

    week_window = Window.partitionBy("wm_yr_wk")

    feature_df = (
        feature_df
        .withColumn(
            "cce_week_avg_log_quantity",
            F.avg("log_quantity").over(week_window)
        )
        .withColumn(
            "cce_week_avg_log_price",
            F.avg("log_price").over(week_window)
        )
    )

    store_week_window = Window.partitionBy("pe_store_group", "wm_yr_wk")

    feature_df = (
        feature_df
        .withColumn(
            "cce_store_week_avg_log_quantity",
            F.avg("log_quantity").over(store_week_window)
        )
        .withColumn(
            "cce_store_week_avg_log_price",
            F.avg("log_price").over(store_week_window)
        )
    )

    if "pe_category" in feature_df.columns:
        category_week_window = Window.partitionBy("pe_category", "wm_yr_wk")

        feature_df = (
            feature_df
            .withColumn(
                "cce_category_week_avg_log_quantity",
                F.avg("log_quantity").over(category_week_window)
            )
            .withColumn(
                "cce_category_week_avg_log_price",
                F.avg("log_price").over(category_week_window)
            )
        )

    if "pe_product_type" in feature_df.columns:
        product_type_week_window = Window.partitionBy("pe_product_type", "wm_yr_wk")

        feature_df = (
            feature_df
            .withColumn(
                "cce_product_type_week_avg_log_quantity",
                F.avg("log_quantity").over(product_type_week_window)
            )
            .withColumn(
                "cce_product_type_week_avg_log_price",
                F.avg("log_price").over(product_type_week_window)
            )
        )

    if "pe_product_division" in feature_df.columns:
        division_week_window = Window.partitionBy("pe_product_division", "wm_yr_wk")

        feature_df = (
            feature_df
            .withColumn(
                "cce_product_division_week_avg_log_quantity",
                F.avg("log_quantity").over(division_week_window)
            )
            .withColumn(
                "cce_product_division_week_avg_log_price",
                F.avg("log_price").over(division_week_window)
            )
        )

    if "pe_country" in feature_df.columns:
        country_week_window = Window.partitionBy("pe_country", "wm_yr_wk")

        feature_df = (
            feature_df
            .withColumn(
                "cce_country_week_avg_log_quantity",
                F.avg("log_quantity").over(country_week_window)
            )
            .withColumn(
                "cce_country_week_avg_log_price",
                F.avg("log_price").over(country_week_window)
            )
        )

    return feature_df


# ============================================================
# 8. Add training-readiness flags
# ============================================================

def add_training_readiness_flags(feature_df):
    """
    Add row-level flags for causal, sales, probability, and MDO usage.
    """

    print("Adding training-readiness flags...")

    group_stats_df = (
        feature_df
        .groupBy("pe_article_store_group")
        .agg(
            F.count("*").alias("rows_per_product_store"),
            F.countDistinct("wm_yr_wk").alias("weeks_per_product_store"),
            F.countDistinct(
                F.when(F.col("valid_price_flag") == 1, F.col("pe_unit_price"))
            ).alias("price_variation_count"),
            F.countDistinct(
                F.when(F.col("valid_price_flag") == 1, F.col("discount"))
            ).alias("discount_variation_count"),
        )
    )

    feature_df = feature_df.join(
        group_stats_df,
        on="pe_article_store_group",
        how="left"
    )

    feature_df = (
        feature_df
        .withColumn(
            "valid_for_sales_training",
            F.when(
                (F.col("valid_for_price_elasticity") == 1) &
                (F.col("pe_quantity") > 0) &
                (F.col("sold_qty_log").isNotNull()) &
                (F.col("log_price").isNotNull()),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
        .withColumn(
            "valid_for_probability_training",
            F.when(
                (F.col("pe_quantity") >= 0) &
                (F.col("probability_target").isNotNull()) &
                (F.col("probability_target_smoothed").isNotNull()) &
                (F.col("probability_logit_target").isNotNull()),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
        .withColumn(
            "valid_for_probability_training_with_price",
            F.when(
                (F.col("valid_for_probability_training") == 1) &
                (F.col("valid_for_price_elasticity") == 1) &
                (F.col("log_price").isNotNull()) &
                (F.col("discount").isNotNull()),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
        .withColumn(
            "valid_for_causal_training",
            F.when(
                (F.col("valid_for_price_elasticity") == 1) &
                (F.col("valid_price_flag") == 1) &
                (F.col("pe_quantity") > 0) &
                (F.col("sold_qty_log").isNotNull()) &
                (F.col("log_price").isNotNull()) &
                (F.col("discount").isNotNull()) &
                (F.col("weeks_per_product_store") >= F.lit(MIN_HISTORY_POINTS_FOR_CAUSAL)) &
                (
                    (F.col("price_variation_count") > 1) |
                    (F.col("discount_variation_count") > 1)
                ),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
    )

    return feature_df


# ============================================================
# 9. Create quality summary
# ============================================================

def create_feature_quality_summary(feature_df):
    quality_df = (
        feature_df
        .groupBy("pe_store_group")
        .agg(
            F.count("*").alias("rows"),
            F.countDistinct("pe_article").alias("products"),
            F.countDistinct("wm_yr_wk").alias("weeks"),
            F.countDistinct("pe_article_store_group").alias("product_store_groups"),

            F.sum("valid_price_flag").alias("rows_with_valid_price"),
            F.sum("valid_for_price_elasticity").alias("rows_valid_for_price_elasticity"),
            F.sum("valid_for_sales_training").alias("rows_valid_for_sales_training"),
            F.sum("valid_for_causal_training").alias("rows_valid_for_causal_training"),
            F.sum("valid_for_probability_training").alias("rows_valid_for_probability_training"),
            F.sum("valid_for_probability_training_with_price").alias("rows_valid_for_probability_training_with_price"),
            F.sum("valid_for_mdo_input").alias("rows_valid_for_mdo_input"),

            F.min("date").alias("min_date"),
            F.max("date").alias("max_date"),

            F.avg("probability_target").alias("avg_probability_target"),
            F.avg("probability_target_smoothed").alias("avg_probability_target_smoothed"),
            F.avg("probability_logit_target").alias("avg_probability_logit_target"),
        )
        .withColumn(
            "causal_training_pct",
            F.col("rows_valid_for_causal_training") / F.col("rows")
        )
        .withColumn(
            "sales_training_pct",
            F.col("rows_valid_for_sales_training") / F.col("rows")
        )
        .withColumn(
            "probability_training_with_price_pct",
            F.col("rows_valid_for_probability_training_with_price") / F.col("rows")
        )
        .withColumn(
            "mdo_ready_pct",
            F.col("rows_valid_for_mdo_input") / F.col("rows")
        )
    )

    return quality_df


# ============================================================
# 10. Runner
# ============================================================

def run_causal_feature_engineering():
    print("==============================================")
    print("Running PE Feature Engineering")
    print("==============================================")

    try:
        user_email = (
            dbutils.notebook.entry_point
            .getDbutils()
            .notebook()
            .getContext()
            .userName()
            .get()
        )

        project_root = f"/Workspace/Users/{user_email}/PE_work"

        if project_root not in sys.path:
            sys.path.append(project_root)

        print("Project root added:", project_root)

    except Exception as exc:
        print("Could not auto-detect project root.")
        print("Error:", exc)

    base_df = load_base_table()

    feature_base_df = prepare_standard_feature_base(base_df)

    feature_df = add_core_model_features(feature_base_df)

    feature_df = add_cce_features(feature_df)

    feature_df = add_training_readiness_flags(feature_df)

    feature_quality_df = create_feature_quality_summary(feature_df)

    print("Saving feature table:", OUTPUT_FEATURE_TABLE)

    (
        feature_df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(OUTPUT_FEATURE_TABLE)
    )

    feature_df.createOrReplaceTempView(OUTPUT_FEATURE_VIEW)

    print("Saving feature quality table:", OUTPUT_FEATURE_QUALITY_TABLE)

    (
        feature_quality_df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(OUTPUT_FEATURE_QUALITY_TABLE)
    )

    print("Temporary view created:", OUTPUT_FEATURE_VIEW)
    print("Permanent feature table saved:", OUTPUT_FEATURE_TABLE)
    print("Feature quality table saved:", OUTPUT_FEATURE_QUALITY_TABLE)

    print("Validation 1: feature table size")
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
            FROM {OUTPUT_FEATURE_TABLE}
        """)
    )

    print("Validation 2: feature quality by store")
    display(
        spark.sql(f"""
            SELECT *
            FROM {OUTPUT_FEATURE_QUALITY_TABLE}
            ORDER BY pe_store_group
        """)
    )

    print("Validation 3: causal training readiness")
    display(
        spark.sql(f"""
            SELECT
                valid_price_flag,
                valid_for_price_elasticity,
                valid_for_causal_training,
                COUNT(*) AS rows
            FROM {OUTPUT_FEATURE_TABLE}
            GROUP BY
                valid_price_flag,
                valid_for_price_elasticity,
                valid_for_causal_training
            ORDER BY
                valid_price_flag,
                valid_for_price_elasticity,
                valid_for_causal_training
        """)
    )

    print("Validation 4: probability logit target check")
    display(
        spark.sql(f"""
            SELECT
                probability_target,
                MIN(probability_target_smoothed) AS min_probability_target_smoothed,
                MAX(probability_target_smoothed) AS max_probability_target_smoothed,
                MIN(probability_logit_target) AS min_probability_logit_target,
                MAX(probability_logit_target) AS max_probability_logit_target,
                COUNT(*) AS rows
            FROM {OUTPUT_FEATURE_TABLE}
            GROUP BY probability_target
            ORDER BY probability_target
        """)
    )

    print("Validation 5: rows per product-store")
    display(
        spark.sql(f"""
            SELECT
                MIN(rows_per_product_store) AS min_rows_per_product_store,
                MAX(rows_per_product_store) AS max_rows_per_product_store,
                AVG(rows_per_product_store) AS avg_rows_per_product_store,
                MIN(weeks_per_product_store) AS min_weeks_per_product_store,
                MAX(weeks_per_product_store) AS max_weeks_per_product_store,
                AVG(weeks_per_product_store) AS avg_weeks_per_product_store
            FROM {OUTPUT_FEATURE_TABLE}
        """)
    )

    print("Validation 6: CCE columns check")
    display(
        spark.sql(f"""
            SELECT
                COUNT(*) AS rows,
                AVG(cce_week_avg_log_quantity) AS avg_cce_week_qty,
                AVG(cce_week_avg_log_price) AS avg_cce_week_price,
                AVG(cce_store_week_avg_log_quantity) AS avg_cce_store_week_qty,
                AVG(cce_store_week_avg_log_price) AS avg_cce_store_week_price
            FROM {OUTPUT_FEATURE_TABLE}
        """)
    )

    print("Sample feature rows")
    display(
        spark.sql(f"""
            SELECT
                date,
                wm_yr_wk,
                pe_article,
                pe_store_group,
                pe_article_store_group,
                pe_quantity,
                pe_unit_price,
                discount,
                valid_price_flag,
                valid_for_price_elasticity,
                valid_for_causal_training,
                valid_for_sales_training,
                valid_for_probability_training,
                valid_for_probability_training_with_price,
                valid_for_mdo_input,
                log_quantity,
                log_price,
                sold_qty_log,
                probability_target,
                probability_target_smoothed,
                probability_logit_target,
                discount_power_1,
                discount_power_2,
                discount_power_3,
                discount_power_4,
                log_store_stock_quantity,
                log_inventory_onhand_quantity,
                cce_week_avg_log_quantity,
                cce_week_avg_log_price,
                cce_store_week_avg_log_quantity,
                cce_store_week_avg_log_price
            FROM {OUTPUT_FEATURE_TABLE}
            ORDER BY pe_store_group, pe_article, wm_yr_wk
            LIMIT 100
        """)
    )

    print("==============================================")
    print("PE Feature Engineering Completed Successfully")
    print("==============================================")

    return feature_df


# ============================================================
# 11. Execute
# ============================================================

causal_features_df = run_causal_feature_engineering()

if __name__ == "__main__":
    run_causal_feature_engineering()