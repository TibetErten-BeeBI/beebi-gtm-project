import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# 1. Spark setup
# ============================================================

spark = SparkSession.builder.getOrCreate()


# ============================================================
# 2. Config - generic table naming
# ============================================================

def get_param(param_name: str, default_value: str = "") -> str:
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


def view_name(base_name: str) -> str:
    if TABLE_PREFIX:
        return f"{TABLE_PREFIX}_{base_name}"
    return base_name


SOURCE_BASE_TABLE = table_name("base_data_table")

OUTPUT_FEATURE_TABLE = table_name("pe_causal_features_dev")
OUTPUT_FEATURE_VIEW = view_name("pe_causal_features_dev_view")

OUTPUT_FEATURE_QUALITY_TABLE = table_name("pe_causal_features_quality_summary")


# Daily version:
# Old weekly code used MIN_HISTORY_POINTS_FOR_CAUSAL = 4 weeks.
# 4 weeks is roughly 28 days.
MIN_HISTORY_POINTS_FOR_CAUSAL_DAYS = int(
    get_param("min_history_points_for_causal_days", "28")
)

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

    if "base_data_grain" in base_df.columns:
        grains = [
            row["base_data_grain"]
            for row in base_df.select("base_data_grain").distinct().collect()
        ]

        print("Base data grain values:", grains)

        if "product_store_day" not in grains:
            raise ValueError(
                "This daily feature engineering file expects base_data_grain = product_store_day. "
                f"Found grains: {grains}. "
                "Run the modified daily base data builder first."
            )

    return base_df


# ============================================================
# 5. Prepare standard feature base
# ============================================================

def prepare_standard_feature_base(base_df):
    """
    Convert base_data_table into a clean product-store-day feature table.

    Grain:
        pe_article + pe_store_group + date
    """

    print("Preparing standard DAILY PE feature base...")

    feature_base_df = (
        base_df
        .select(
            F.to_date(F.col("date")).alias("date"),
            safe_select_col(base_df, "week_start_date", "week_start_date"),
            F.col("wm_yr_wk").cast("int").alias("wm_yr_wk"),

            safe_select_col(base_df, "calendar_month", "calendar_month", "int"),
            safe_select_col(base_df, "calendar_year", "calendar_year", "int"),
            safe_select_col(base_df, "calendar_day_id", "calendar_day_id"),
            safe_select_col(base_df, "weekday", "weekday"),
            safe_select_col(base_df, "wday", "wday", "int"),

            F.col("pe_article").alias("pe_article"),
            F.col("pe_store_group").alias("pe_store_group"),
            safe_select_col(base_df, "pe_article_store_group", "pe_article_store_group"),

            F.col("pe_quantity").cast("double").alias("pe_quantity"),
            safe_select_col(base_df, "days_in_week", "days_in_week", "int", 1),
            safe_select_col(base_df, "days_sold_count", "days_sold_count", "int", 0),

            safe_select_col(base_df, "pe_unit_price", "pe_unit_price", "double"),
            safe_select_col(base_df, "pe_actual_retail_price", "pe_actual_retail_price", "double"),
            safe_select_col(base_df, "pe_average_zone_retail_price", "pe_average_zone_retail_price", "double"),
            safe_select_col(base_df, "discount", "discount", "double"),
            safe_select_col(base_df, "pe_sales_amount", "pe_sales_amount", "double"),

            safe_select_col(base_df, "pe_store_stock_quantity", "pe_store_stock_quantity", "double"),
            safe_select_col(base_df, "pe_inventory_onhand_quantity", "pe_inventory_onhand_quantity", "double"),
            safe_select_col(base_df, "pe_store_stock_quantity_for_model", "pe_store_stock_quantity_for_model", "double", 0.0),
            safe_select_col(base_df, "pe_inventory_onhand_quantity_for_model", "pe_inventory_onhand_quantity_for_model", "double", 0.0),

            safe_select_col(base_df, "stock_date", "stock_date"),
            safe_select_col(base_df, "stock_start_date", "stock_start_date"),
            safe_select_col(base_df, "stock_end_date", "stock_end_date"),
            safe_select_col(base_df, "inventory_date", "inventory_date"),
            safe_select_col(base_df, "inventory_start_date", "inventory_start_date"),
            safe_select_col(base_df, "inventory_end_date", "inventory_end_date"),

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

    feature_base_df = (
        feature_base_df
        .withColumn(
            "day_of_week",
            F.coalesce(F.col("wday").cast("int"), F.dayofweek(F.col("date")))
        )
        .withColumn("day_of_month", F.dayofmonth(F.col("date")))
        .withColumn("week_of_year", F.weekofyear(F.col("date")))
        .withColumn("is_weekend", F.when(F.col("day_of_week").isin(1, 7), F.lit(1)).otherwise(F.lit(0)))
    )

    print("Standard DAILY feature base rows:", feature_base_df.count())

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
        .withColumn("discount_power_1", F.col("discount").cast("double"))
        .withColumn("discount_power_2", F.when(F.col("discount").isNotNull(), F.pow(F.col("discount"), 2)))
        .withColumn("discount_power_3", F.when(F.col("discount").isNotNull(), F.pow(F.col("discount"), 3)))
        .withColumn("discount_power_4", F.when(F.col("discount").isNotNull(), F.pow(F.col("discount"), 4)))
    )

    return feature_df


# ============================================================
# 7. Add CCE features
# ============================================================

def add_cce_features(feature_df):
    print("Adding DAILY CCE features...")

    day_window = Window.partitionBy("date")

    feature_df = (
        feature_df
        .withColumn("cce_day_avg_log_quantity", F.avg("log_quantity").over(day_window))
        .withColumn("cce_day_avg_log_price", F.avg("log_price").over(day_window))
    )

    store_day_window = Window.partitionBy("pe_store_group", "date")

    feature_df = (
        feature_df
        .withColumn("cce_store_day_avg_log_quantity", F.avg("log_quantity").over(store_day_window))
        .withColumn("cce_store_day_avg_log_price", F.avg("log_price").over(store_day_window))
    )

    if "pe_category" in feature_df.columns:
        category_day_window = Window.partitionBy("pe_category", "date")
        feature_df = (
            feature_df
            .withColumn("cce_category_day_avg_log_quantity", F.avg("log_quantity").over(category_day_window))
            .withColumn("cce_category_day_avg_log_price", F.avg("log_price").over(category_day_window))
        )

    if "pe_product_type" in feature_df.columns:
        product_type_day_window = Window.partitionBy("pe_product_type", "date")
        feature_df = (
            feature_df
            .withColumn("cce_product_type_day_avg_log_quantity", F.avg("log_quantity").over(product_type_day_window))
            .withColumn("cce_product_type_day_avg_log_price", F.avg("log_price").over(product_type_day_window))
        )

    if "pe_product_division" in feature_df.columns:
        division_day_window = Window.partitionBy("pe_product_division", "date")
        feature_df = (
            feature_df
            .withColumn("cce_product_division_day_avg_log_quantity", F.avg("log_quantity").over(division_day_window))
            .withColumn("cce_product_division_day_avg_log_price", F.avg("log_price").over(division_day_window))
        )

    if "pe_country" in feature_df.columns:
        country_day_window = Window.partitionBy("pe_country", "date")
        feature_df = (
            feature_df
            .withColumn("cce_country_day_avg_log_quantity", F.avg("log_quantity").over(country_day_window))
            .withColumn("cce_country_day_avg_log_price", F.avg("log_price").over(country_day_window))
        )

    week_window = Window.partitionBy("wm_yr_wk")

    feature_df = (
        feature_df
        .withColumn("cce_week_avg_log_quantity", F.avg("log_quantity").over(week_window))
        .withColumn("cce_week_avg_log_price", F.avg("log_price").over(week_window))
    )

    store_week_window = Window.partitionBy("pe_store_group", "wm_yr_wk")

    feature_df = (
        feature_df
        .withColumn("cce_store_week_avg_log_quantity", F.avg("log_quantity").over(store_week_window))
        .withColumn("cce_store_week_avg_log_price", F.avg("log_price").over(store_week_window))
    )

    if "pe_category" in feature_df.columns:
        category_week_window = Window.partitionBy("pe_category", "wm_yr_wk")
        feature_df = (
            feature_df
            .withColumn("cce_category_week_avg_log_quantity", F.avg("log_quantity").over(category_week_window))
            .withColumn("cce_category_week_avg_log_price", F.avg("log_price").over(category_week_window))
        )

    if "pe_product_type" in feature_df.columns:
        product_type_week_window = Window.partitionBy("pe_product_type", "wm_yr_wk")
        feature_df = (
            feature_df
            .withColumn("cce_product_type_week_avg_log_quantity", F.avg("log_quantity").over(product_type_week_window))
            .withColumn("cce_product_type_week_avg_log_price", F.avg("log_price").over(product_type_week_window))
        )

    if "pe_product_division" in feature_df.columns:
        division_week_window = Window.partitionBy("pe_product_division", "wm_yr_wk")
        feature_df = (
            feature_df
            .withColumn("cce_product_division_week_avg_log_quantity", F.avg("log_quantity").over(division_week_window))
            .withColumn("cce_product_division_week_avg_log_price", F.avg("log_price").over(division_week_window))
        )

    if "pe_country" in feature_df.columns:
        country_week_window = Window.partitionBy("pe_country", "wm_yr_wk")
        feature_df = (
            feature_df
            .withColumn("cce_country_week_avg_log_quantity", F.avg("log_quantity").over(country_week_window))
            .withColumn("cce_country_week_avg_log_price", F.avg("log_price").over(country_week_window))
        )

    return feature_df


# ============================================================
# 8. Add training-readiness flags + Phase 2 MDO support columns
# ============================================================

def add_training_readiness_flags(feature_df):
    """
    Add row-level flags for causal, sales, probability, and MDO usage.

    Phase 2 additions:
    - history_days
    - price_variation
    - discount_variation
    - prediction_confidence
    """

    print("Adding DAILY training-readiness flags and Phase 2 MDO support columns...")

    group_stats_df = (
        feature_df
        .groupBy("pe_article_store_group")
        .agg(
            F.count("*").alias("rows_per_product_store"),
            F.countDistinct("date").alias("days_per_product_store"),
            F.countDistinct("wm_yr_wk").alias("weeks_per_product_store"),
            F.countDistinct(
                F.when(F.col("valid_price_flag") == 1, F.col("pe_unit_price"))
            ).alias("price_variation_count"),
            F.countDistinct(
                F.when(F.col("valid_price_flag") == 1, F.col("discount"))
            ).alias("discount_variation_count"),

            # Rule 31 - Price Variation Rule:
            # Calculate price standard deviation so MDO can detect low price movement.
            F.stddev(
                F.when(F.col("valid_price_flag") == 1, F.col("pe_unit_price"))
            ).alias("price_variation"),

            # Rule 32 - Discount Variation Rule:
            # Calculate discount standard deviation so MDO can detect low discount movement.
            F.stddev(
                F.when(F.col("valid_price_flag") == 1, F.col("discount"))
            ).alias("discount_variation"),
        )
    )

    feature_df = feature_df.join(
        group_stats_df,
        on="pe_article_store_group",
        how="left"
    )

    feature_df = (
        feature_df

        # Rule 30 - Minimum History Rule:
        # Use number of available product-store days as history_days for Phase 2 MDO rule.
        .withColumn(
            "history_days",
            F.coalesce(F.col("days_per_product_store").cast("double"), F.lit(0.0))
        )

        # Rule 31 - Price Variation Rule:
        # Use price standard deviation; if null, set 0 so low-variation rows can be detected.
        .withColumn(
            "price_variation",
            F.coalesce(F.col("price_variation").cast("double"), F.lit(0.0))
        )

        # Rule 32 - Discount Variation Rule:
        # Use discount standard deviation; if null, set 0 so low-variation rows can be detected.
        .withColumn(
            "discount_variation",
            F.coalesce(F.col("discount_variation").cast("double"), F.lit(0.0))
        )

        # Rule 33 - Low Confidence Prediction Rule:
        # Feature engineering does not produce model confidence yet, so set null.
        # MDO will apply this rule only when prediction_confidence is available from model output.
        .withColumn(
            "prediction_confidence",
            F.lit(None).cast("double")
        )

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
                (F.col("days_per_product_store") >= F.lit(MIN_HISTORY_POINTS_FOR_CAUSAL_DAYS)) &
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
            F.countDistinct("date").alias("days"),
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

            F.avg("history_days").alias("avg_history_days"),
            F.avg("price_variation").alias("avg_price_variation"),
            F.avg("discount_variation").alias("avg_discount_variation"),

            F.avg("probability_target").alias("avg_probability_target"),
            F.avg("probability_target_smoothed").alias("avg_probability_target_smoothed"),
            F.avg("probability_logit_target").alias("avg_probability_logit_target"),
        )
        .withColumn("causal_training_pct", F.col("rows_valid_for_causal_training") / F.col("rows"))
        .withColumn("sales_training_pct", F.col("rows_valid_for_sales_training") / F.col("rows"))
        .withColumn("probability_training_with_price_pct", F.col("rows_valid_for_probability_training_with_price") / F.col("rows"))
        .withColumn("mdo_ready_pct", F.col("rows_valid_for_mdo_input") / F.col("rows"))
    )

    return quality_df


# ============================================================
# 10. Runner
# ============================================================

def run_causal_feature_engineering():
    print("==============================================")
    print("Running DAILY PE Feature Engineering")
    print("==============================================")
    print("Source base table:", SOURCE_BASE_TABLE)
    print("Output feature table:", OUTPUT_FEATURE_TABLE)
    print("Minimum daily history for causal:", MIN_HISTORY_POINTS_FOR_CAUSAL_DAYS)

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

    print("Validation: Phase 2 MDO support columns")
    display(
        spark.sql(f"""
            SELECT
                COUNT(*) AS rows,
                MIN(history_days) AS min_history_days,
                MAX(history_days) AS max_history_days,
                AVG(history_days) AS avg_history_days,
                MIN(price_variation) AS min_price_variation,
                MAX(price_variation) AS max_price_variation,
                AVG(price_variation) AS avg_price_variation,
                MIN(discount_variation) AS min_discount_variation,
                MAX(discount_variation) AS max_discount_variation,
                AVG(discount_variation) AS avg_discount_variation
            FROM {OUTPUT_FEATURE_TABLE}
        """)
    )

    print("==============================================")
    print("DAILY PE Feature Engineering Completed Successfully")
    print("==============================================")

    return feature_df


# ============================================================
# 11. Execute
# ============================================================

if __name__ == "__main__":
    causal_features_df = run_causal_feature_engineering()