import re
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# 1. Spark setup
# ============================================================

spark = SparkSession.builder.getOrCreate()


# ============================================================
# 2. Table configuration
# ============================================================

CATALOG_SCHEMA = "workspace.default"

OUTPUT_BASE_TABLE = f"{CATALOG_SCHEMA}.base_data_table"
OUTPUT_QUALITY_TABLE = f"{CATALOG_SCHEMA}.base_data_quality_summary"

CALENDAR_TABLE = f"{CATALOG_SCHEMA}.calendar"
STORE_TABLE = f"{CATALOG_SCHEMA}.dim_pe_store_types"
ARTICLE_LIST_TABLE = f"{CATALOG_SCHEMA}.dim_pov_article_list"
PRODUCT_TABLE = f"{CATALOG_SCHEMA}.dim_product_pd_group_article"

SELLOUT_TABLE = f"{CATALOG_SCHEMA}.mdo_retail_sellout_vw"
PRICE_TABLE = f"{CATALOG_SCHEMA}.mdo_retail_price_history_table"
STOCK_TABLE = f"{CATALOG_SCHEMA}.mdo_retail_stock_vw"
INVENTORY_TABLE = f"{CATALOG_SCHEMA}.inventory_storemovements_tracki"


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


def first_available_column(df, possible_names):
    for col_name in possible_names:
        if col_name in df.columns:
            return col_name
    return None


def required_column(df, possible_names, table_label):
    selected_col = first_available_column(df, possible_names)

    if selected_col is None:
        raise ValueError(
            f"Missing required column in {table_label}. "
            f"Expected one of {possible_names}. "
            f"Available columns: {df.columns}"
        )

    return selected_col


def safe_column(df, source_col, alias_col=None, cast_type=None):
    alias_col = alias_col or source_col

    if source_col in df.columns:
        expr = F.col(source_col)

        if cast_type:
            expr = expr.cast(cast_type)

        return expr.alias(alias_col)

    if cast_type:
        return F.lit(None).cast(cast_type).alias(alias_col)

    return F.lit(None).alias(alias_col)


def find_daily_sales_columns(df):
    return [
        col_name
        for col_name in df.columns
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", col_name)
    ]


def deduplicate_by_latest(df, key_cols, order_cols):
    available_order_cols = [
        col_name
        for col_name in order_cols
        if col_name in df.columns
    ]

    if not available_order_cols:
        return df.dropDuplicates(key_cols)

    window_spec = Window.partitionBy(*key_cols).orderBy(
        *[
            F.col(col_name).desc_nulls_last()
            for col_name in available_order_cols
        ]
    )

    return (
        df
        .withColumn("_row_number", F.row_number().over(window_spec))
        .filter(F.col("_row_number") == 1)
        .drop("_row_number")
    )


# ============================================================
# 4. Calendar
# ============================================================

def load_calendar():
    require_table(CALENDAR_TABLE)

    raw_df = spark.table(CALENDAR_TABLE)

    calendar_df = (
        raw_df
        .select(
            F.to_date(F.col("date")).alias("calendar_date"),
            F.col("wm_yr_wk").cast("int").alias("wm_yr_wk"),

            safe_column(raw_df, "weekday", "weekday"),
            safe_column(raw_df, "wday", "wday", "int"),
            safe_column(raw_df, "month", "calendar_month", "int"),
            safe_column(raw_df, "year", "calendar_year", "int"),
            safe_column(raw_df, "d", "calendar_day_id"),

            safe_column(raw_df, "event_name_1", "event_name_1"),
            safe_column(raw_df, "event_type_1", "event_type_1"),
            safe_column(raw_df, "event_name_2", "event_name_2"),
            safe_column(raw_df, "event_type_2", "event_type_2"),

            safe_column(raw_df, "snap_CA", "snap_CA", "int"),
            safe_column(raw_df, "snap_TX", "snap_TX", "int"),
            safe_column(raw_df, "snap_WI", "snap_WI", "int"),
        )
        .filter(F.col("calendar_date").isNotNull())
        .filter(F.col("wm_yr_wk").isNotNull())
        .dropDuplicates(["calendar_date"])
    )

    print("Calendar rows:", calendar_df.count())

    return calendar_df


# ============================================================
# 5. Sellout: wide daily sales columns to product-store-week
# ============================================================

def build_weekly_sellout(calendar_df):
    require_table(SELLOUT_TABLE)

    raw_df = spark.table(SELLOUT_TABLE)

    article_col = required_column(
        raw_df,
        ["group_article_id", "article_id"],
        "sellout"
    )

    store_col = required_column(
        raw_df,
        ["store_pos", "reporting_unit_group", "reporting_unit", "store_number"],
        "sellout"
    )

    daily_cols = find_daily_sales_columns(raw_df)

    if not daily_cols:
        raise ValueError(
            "No daily sales columns found in sellout table. "
            "Expected columns like 2015-04-26."
        )

    print("Daily sellout columns found:", len(daily_cols))

    stack_expression = "stack({0}, {1}) as (calendar_date, daily_quantity)".format(
        len(daily_cols),
        ", ".join([f"'{col_name}', `{col_name}`" for col_name in daily_cols])
    )

    daily_df = (
        raw_df
        .select(
            F.col(article_col).alias("pe_article"),
            F.col(store_col).alias("pe_store_group"),

            safe_column(raw_df, "product_group", "sellout_product_group"),
            safe_column(raw_df, "country", "pe_country"),
            safe_column(raw_df, "row_hash_id", "sellout_row_hash_id"),
            safe_column(raw_df, "batch_id", "sellout_batch_id"),
            safe_column(raw_df, "project_id", "sellout_project_id"),

            safe_column(raw_df, "sold_quantity", "source_total_sold_quantity", "double"),
            safe_column(raw_df, "net_quantity_sales_unit_of_measure", "source_total_net_quantity", "double"),
            safe_column(raw_df, "return_quantity", "source_total_return_quantity", "double"),
            safe_column(
                raw_df,
                "net_sales_gross_value_added_tax_amount_current_document",
                "source_net_sales_amount",
                "double"
            ),

            F.expr(stack_expression)
        )
        .withColumn("calendar_date", F.to_date(F.col("calendar_date")))
        .withColumn(
            "daily_quantity",
            F.coalesce(F.col("daily_quantity").cast("double"), F.lit(0.0))
        )
        .filter(F.col("pe_article").isNotNull())
        .filter(F.col("pe_store_group").isNotNull())
        .filter(F.col("calendar_date").isNotNull())
    )

    daily_with_calendar_df = (
        daily_df
        .join(calendar_df, on="calendar_date", how="left")
        .filter(F.col("wm_yr_wk").isNotNull())
    )

    weekly_df = (
        daily_with_calendar_df
        .groupBy(
            "pe_article",
            "pe_store_group",
            "wm_yr_wk"
        )
        .agg(
            F.min("calendar_date").alias("week_start_date"),
            F.max("calendar_date").alias("date"),

            F.sum("daily_quantity").alias("pe_quantity"),
            F.count("*").alias("days_in_week"),
            F.sum(
                F.when(F.col("daily_quantity") > 0, F.lit(1)).otherwise(F.lit(0))
            ).alias("days_sold_count"),

            F.first("sellout_product_group", ignorenulls=True).alias("sellout_product_group"),
            F.first("pe_country", ignorenulls=True).alias("pe_country"),
            F.first("sellout_row_hash_id", ignorenulls=True).alias("sellout_row_hash_id"),
            F.first("sellout_batch_id", ignorenulls=True).alias("sellout_batch_id"),
            F.first("sellout_project_id", ignorenulls=True).alias("sellout_project_id"),

            F.first("source_total_sold_quantity", ignorenulls=True).alias("source_total_sold_quantity"),
            F.first("source_total_net_quantity", ignorenulls=True).alias("source_total_net_quantity"),
            F.first("source_total_return_quantity", ignorenulls=True).alias("source_total_return_quantity"),
            F.first("source_net_sales_amount", ignorenulls=True).alias("source_net_sales_amount"),

            F.max("calendar_month").alias("calendar_month"),
            F.max("calendar_year").alias("calendar_year"),
            F.max("event_name_1").alias("event_name_1"),
            F.max("event_type_1").alias("event_type_1"),
            F.max("event_name_2").alias("event_name_2"),
            F.max("event_type_2").alias("event_type_2"),
            F.max("snap_CA").alias("snap_CA"),
            F.max("snap_TX").alias("snap_TX"),
            F.max("snap_WI").alias("snap_WI"),
        )
    )

    print("Weekly sellout rows:", weekly_df.count())

    return weekly_df


# ============================================================
# 6. Dimensions
# ============================================================

def load_article_list():
    require_table(ARTICLE_LIST_TABLE)

    raw_df = spark.table(ARTICLE_LIST_TABLE)

    article_col = required_column(
        raw_df,
        ["group_article_id"],
        "article list"
    )

    article_df = (
        raw_df
        .select(
            F.col(article_col).alias("pe_article"),
            safe_column(raw_df, "row_surrogate_id", "article_row_surrogate_id", "long"),
            safe_column(raw_df, "batch_id", "article_batch_id"),
            safe_column(raw_df, "project_id", "article_project_id"),
        )
        .dropDuplicates(["pe_article"])
    )

    print("Article list rows:", article_df.count())

    return article_df


def load_product_attributes():
    require_table(PRODUCT_TABLE)

    raw_df = spark.table(PRODUCT_TABLE)

    article_col = required_column(
        raw_df,
        ["group_article_id"],
        "product"
    )

    product_df = (
        raw_df
        .select(
            F.col(article_col).alias("pe_article"),
            safe_column(raw_df, "row_surrogate_id", "product_row_surrogate_id", "long"),
            safe_column(raw_df, "batch_id", "product_batch_id"),
            safe_column(raw_df, "project_id", "product_project_id"),

            safe_column(raw_df, "merchandise_gender_description", "pe_gender"),
            safe_column(raw_df, "merchandise_category_description", "pe_category"),
            safe_column(raw_df, "merchandise_product_type_description", "pe_product_type"),
            safe_column(raw_df, "merchandise_product_division_description", "pe_product_division"),
        )
        .dropDuplicates(["pe_article"])
    )

    print("Product attribute rows:", product_df.count())

    return product_df


def load_store_attributes():
    require_table(STORE_TABLE)

    raw_df = spark.table(STORE_TABLE)

    store_col = required_column(
        raw_df,
        ["store_number"],
        "store"
    )

    store_df = (
        raw_df
        .select(
            F.col(store_col).alias("pe_store_group"),
            safe_column(raw_df, "row_surrogate_id", "store_row_surrogate_id", "long"),
            safe_column(raw_df, "batch_id", "store_batch_id"),
            safe_column(raw_df, "project_id", "store_project_id"),
            safe_column(raw_df, "updated_date", "store_updated_date"),
        )
        .dropDuplicates(["pe_store_group"])
        .withColumn("pe_store_state", F.split(F.col("pe_store_group"), "_").getItem(0))
        .withColumn("pe_store_type", F.split(F.col("pe_store_group"), "_").getItem(0))
    )

    print("Store rows:", store_df.count())

    return store_df


# ============================================================
# 7. Price, stock, inventory
# ============================================================

def load_price_history():
    require_table(PRICE_TABLE)

    raw_df = spark.table(PRICE_TABLE)

    article_col = required_column(
        raw_df,
        ["group_article_id"],
        "price"
    )

    store_col = required_column(
        raw_df,
        ["reporting_unit", "store_pos", "reporting_unit_group"],
        "price"
    )

    actual_price_col = required_column(
        raw_df,
        ["actual_retail_price", "arp"],
        "price"
    )

    zone_price_col = required_column(
        raw_df,
        ["average_zone_retail_price", "azrp"],
        "price"
    )

    start_col = required_column(
        raw_df,
        ["start_date", "startdate"],
        "price"
    )

    end_col = required_column(
        raw_df,
        ["end_date", "enddate"],
        "price"
    )

    price_df = (
        raw_df
        .select(
            F.col(article_col).alias("price_article"),
            F.col(store_col).alias("price_store"),

            safe_column(raw_df, "row_hash_id", "price_row_hash_id"),
            safe_column(raw_df, "batch_id", "price_batch_id"),
            safe_column(raw_df, "project_id", "price_project_id"),
            safe_column(raw_df, "sales_organization", "price_sales_organization"),

            F.col(actual_price_col).cast("double").alias("raw_actual_retail_price"),
            F.col(zone_price_col).cast("double").alias("raw_average_zone_retail_price"),
            F.to_date(F.col(start_col)).alias("price_start_date"),
            F.to_date(F.col(end_col)).alias("price_end_date"),

            safe_column(raw_df, "data_quality_flag", "price_data_quality_flag"),
        )
        .filter(F.col("price_article").isNotNull())
        .filter(F.col("price_store").isNotNull())
        .filter(F.col("price_start_date").isNotNull())
        .filter(F.col("price_end_date").isNotNull())
    )

    price_df = deduplicate_by_latest(
        price_df,
        key_cols=[
            "price_article",
            "price_store",
            "price_start_date",
            "price_end_date",
        ],
        order_cols=[
            "price_start_date",
            "price_end_date",
        ],
    )

    print("Price rows:", price_df.count())

    return price_df


def load_stock_history():
    require_table(STOCK_TABLE)

    raw_df = spark.table(STOCK_TABLE)

    article_col = required_column(
        raw_df,
        ["group_article_id"],
        "stock"
    )

    store_col = required_column(
        raw_df,
        ["store_pos", "reporting_unit"],
        "stock"
    )

    date_col = required_column(
        raw_df,
        ["snapshot_date"],
        "stock"
    )

    qty_col = required_column(
        raw_df,
        ["store_stock_quantity_base_unit_of_measure"],
        "stock"
    )

    stock_df = (
        raw_df
        .select(
            F.col(article_col).alias("stock_article"),
            F.col(store_col).alias("stock_store"),
            F.to_date(F.col(date_col)).alias("stock_date"),
            F.col(qty_col).cast("double").alias("raw_store_stock_quantity"),

            safe_column(raw_df, "row_hash_id", "stock_row_hash_id"),
            safe_column(raw_df, "batch_id", "stock_batch_id"),
            safe_column(raw_df, "project_id", "stock_project_id"),
            safe_column(raw_df, "country_name", "stock_country_name"),
            safe_column(raw_df, "sales_group", "stock_sales_group"),
            safe_column(raw_df, "distribution_channel", "stock_distribution_channel"),
        )
        .filter(F.col("stock_article").isNotNull())
        .filter(F.col("stock_store").isNotNull())
        .filter(F.col("stock_date").isNotNull())
    )

    stock_df = deduplicate_by_latest(
        stock_df,
        key_cols=[
            "stock_article",
            "stock_store",
            "stock_date",
        ],
        order_cols=[
            "stock_date",
        ],
    )

    print("Stock rows:", stock_df.count())

    return stock_df


def load_inventory_history():
    require_table(INVENTORY_TABLE)

    raw_df = spark.table(INVENTORY_TABLE)

    article_col = required_column(
        raw_df,
        ["article_id", "group_article_id"],
        "inventory"
    )

    store_col = required_column(
        raw_df,
        ["store_pos", "storepos", "reporting_unit_group", "reporting_unit"],
        "inventory"
    )

    date_col = required_column(
        raw_df,
        ["value_date", "valuedate"],
        "inventory"
    )

    qty_col = required_column(
        raw_df,
        ["on_hand_stock_quantity", "onhandstockqty"],
        "inventory"
    )

    inventory_df = (
        raw_df
        .select(
            F.col(article_col).alias("inventory_article"),
            F.col(store_col).alias("inventory_store"),
            F.to_date(F.col(date_col)).alias("inventory_date"),
            F.col(qty_col).cast("double").alias("raw_inventory_onhand_quantity"),

            safe_column(raw_df, "row_hash_id", "inventory_row_hash_id"),
            safe_column(raw_df, "batch_id", "inventory_batch_id"),
            safe_column(raw_df, "project_id", "inventory_project_id"),
            safe_column(raw_df, "country_name", "inventory_country_name"),
            safe_column(raw_df, "sales_group", "inventory_sales_group"),
            safe_column(raw_df, "distribution_channel", "inventory_distribution_channel"),
        )
        .filter(F.col("inventory_article").isNotNull())
        .filter(F.col("inventory_store").isNotNull())
        .filter(F.col("inventory_date").isNotNull())
    )

    inventory_df = deduplicate_by_latest(
        inventory_df,
        key_cols=[
            "inventory_article",
            "inventory_store",
            "inventory_date",
        ],
        order_cols=[
            "inventory_date",
        ],
    )

    print("Inventory rows:", inventory_df.count())

    return inventory_df


# ============================================================
# 8. Final base table
# ============================================================

def create_base_data_table():
    print("====================================================")
    print("Building base_data_table at product-store-week grain")
    print("Long-run version: keep all stores and flag missing source data")
    print("====================================================")

    calendar_df = load_calendar()
    weekly_sellout_df = build_weekly_sellout(calendar_df)

    article_df = load_article_list()
    product_df = load_product_attributes()
    store_df = load_store_attributes()

    price_df = load_price_history()
    stock_df = load_stock_history()
    inventory_df = load_inventory_history()

    print("Joining article, product, and store attributes.")

    base_df = (
        weekly_sellout_df
        .join(article_df, on="pe_article", how="inner")
        .join(product_df, on="pe_article", how="left")
        .join(store_df, on="pe_store_group", how="inner")
    )

    print("Joining price history. Missing price rows will be kept and flagged.")

    base_df = (
        base_df.alias("b")
        .join(
            price_df.alias("p"),
            on=[
                F.col("b.pe_article") == F.col("p.price_article"),
                F.col("b.pe_store_group") == F.col("p.price_store"),
                F.col("b.week_start_date") <= F.col("p.price_end_date"),
                F.col("b.date") >= F.col("p.price_start_date"),
            ],
            how="left"
        )
    )

    price_window = Window.partitionBy(
        "pe_article",
        "pe_store_group",
        "wm_yr_wk"
    ).orderBy(
        F.when(F.col("price_data_quality_flag") == "MAPPED", F.lit(1)).otherwise(F.lit(0)).desc(),
        F.col("price_start_date").desc_nulls_last(),
        F.col("price_end_date").desc_nulls_last()
    )

    base_df = (
        base_df
        .withColumn("_price_rank", F.row_number().over(price_window))
        .filter(F.col("_price_rank") == 1)
        .drop("_price_rank", "price_article", "price_store")
    )

    print("Joining stock. Missing stock rows will be kept and flagged.")

    base_df = (
        base_df.alias("b")
        .join(
            stock_df.alias("s"),
            on=[
                F.col("b.pe_article") == F.col("s.stock_article"),
                F.col("b.pe_store_group") == F.col("s.stock_store"),
                F.col("b.date") == F.col("s.stock_date"),
            ],
            how="left"
        )
        .drop("stock_article", "stock_store")
    )

    print("Joining inventory. Missing inventory rows will be kept and flagged.")

    base_df = (
        base_df.alias("b")
        .join(
            inventory_df.alias("i"),
            on=[
                F.col("b.pe_article") == F.col("i.inventory_article"),
                F.col("b.pe_store_group") == F.col("i.inventory_store"),
                F.col("b.date") == F.col("i.inventory_date"),
            ],
            how="left"
        )
        .drop("inventory_article", "inventory_store")
    )

    print("Creating PE and MDO fields.")

    base_df = (
        base_df
        .withColumn("base_data_grain", F.lit("product_store_week"))
        .withColumn(
            "base_data_hash_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.col("pe_article"),
                    F.col("pe_store_group"),
                    F.col("wm_yr_wk").cast("string")
                ),
                256
            )
        )
        .withColumn(
            "pe_article_store_group",
            F.concat_ws("_", F.col("pe_article"), F.col("pe_store_group"))
        )

        # Price fields
        .withColumn(
            "price_available_flag",
            F.when(
                (F.col("raw_actual_retail_price").isNotNull()) |
                (F.col("raw_average_zone_retail_price").isNotNull()),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
        .withColumn("pe_actual_retail_price", F.col("raw_actual_retail_price"))
        .withColumn("pe_average_zone_retail_price", F.col("raw_average_zone_retail_price"))
        .withColumn(
            "pe_unit_price",
            F.coalesce(
                F.col("pe_actual_retail_price"),
                F.col("pe_average_zone_retail_price")
            )
        )
        .withColumn(
            "valid_price_flag",
            F.when(
                (F.col("price_data_quality_flag") == "MAPPED") &
                (F.col("pe_unit_price").isNotNull()) &
                (F.col("pe_unit_price") > 0),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
        .withColumn(
            "discount",
            F.when(
                (F.col("valid_price_flag") == 1) &
                (F.col("pe_average_zone_retail_price").isNotNull()) &
                (F.col("pe_actual_retail_price").isNotNull()) &
                (F.col("pe_average_zone_retail_price") > 0) &
                (F.col("pe_actual_retail_price") < F.col("pe_average_zone_retail_price")),
                (
                    F.col("pe_average_zone_retail_price") -
                    F.col("pe_actual_retail_price")
                ) / F.col("pe_average_zone_retail_price")
            )
            .when(F.col("valid_price_flag") == 1, F.lit(0.0))
            .otherwise(F.lit(None).cast("double"))
        )
        .withColumn(
            "discount",
            F.when(F.col("discount") < 0, F.lit(0.0))
             .when(F.col("discount") > 0.80, F.lit(0.80))
             .otherwise(F.col("discount"))
        )

        # Stock and inventory fields
        .withColumn(
            "stock_available_flag",
            F.when(F.col("raw_store_stock_quantity").isNotNull(), F.lit(1)).otherwise(F.lit(0))
        )
        .withColumn(
            "inventory_available_flag",
            F.when(F.col("raw_inventory_onhand_quantity").isNotNull(), F.lit(1)).otherwise(F.lit(0))
        )
        .withColumn("pe_store_stock_quantity", F.col("raw_store_stock_quantity"))
        .withColumn("pe_inventory_onhand_quantity", F.col("raw_inventory_onhand_quantity"))

        # Helper columns for ML models only.
        # Raw stock/inventory remain NULL when missing.
        .withColumn(
            "pe_store_stock_quantity_for_model",
            F.coalesce(F.col("raw_store_stock_quantity"), F.lit(0.0))
        )
        .withColumn(
            "pe_inventory_onhand_quantity_for_model",
            F.coalesce(F.col("raw_inventory_onhand_quantity"), F.lit(0.0))
        )

        # Sales amount stays NULL when price is unknown.
        .withColumn(
            "pe_sales_amount",
            F.when(
                F.col("pe_unit_price").isNotNull(),
                F.col("pe_quantity") * F.col("pe_unit_price")
            ).otherwise(F.lit(None).cast("double"))
        )

        # Probability target
        .withColumn(
            "is_sold",
            F.when(F.col("pe_quantity") > 0, F.lit(1)).otherwise(F.lit(0))
        )
        .withColumn("probability_target", F.col("is_sold"))

        # SNAP flag based on state
        .withColumn(
            "snap",
            F.when(F.col("pe_store_state") == "CA", F.col("snap_CA"))
             .when(F.col("pe_store_state") == "TX", F.col("snap_TX"))
             .when(F.col("pe_store_state") == "WI", F.col("snap_WI"))
             .otherwise(F.lit(0))
        )

        # Quality flags for downstream PE/MDO
        .withColumn(
            "valid_for_price_elasticity",
            F.when(
                (F.col("valid_price_flag") == 1) &
                (F.col("price_available_flag") == 1),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
        .withColumn(
            "valid_for_mdo_input",
            F.when(
                (F.col("valid_price_flag") == 1) &
                (F.col("stock_available_flag") == 1) &
                (F.col("inventory_available_flag") == 1),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
        .withColumn(
            "missing_price_reason",
            F.when(F.col("price_available_flag") == 0, F.lit("missing_price_source"))
             .when(F.col("valid_price_flag") == 0, F.lit("invalid_price_source"))
             .otherwise(F.lit(None))
        )
        .withColumn(
            "missing_stock_reason",
            F.when(F.col("stock_available_flag") == 0, F.lit("missing_stock_source"))
             .otherwise(F.lit(None))
        )
        .withColumn(
            "missing_inventory_reason",
            F.when(F.col("inventory_available_flag") == 0, F.lit("missing_inventory_source"))
             .otherwise(F.lit(None))
        )

        # Do not filter on price/stock/inventory here.
        # Keep all product-store-week rows from sellout.
        .filter(F.col("pe_article").isNotNull())
        .filter(F.col("pe_store_group").isNotNull())
        .filter(F.col("wm_yr_wk").isNotNull())
        .filter(F.col("date").isNotNull())
        .filter(F.col("pe_quantity").isNotNull())
        .filter(F.col("pe_quantity") >= 0)
        .dropDuplicates(["pe_article", "pe_store_group", "wm_yr_wk"])
    )

    selected_columns = [
        "base_data_hash_id",
        "base_data_grain",

        "date",
        "week_start_date",
        "wm_yr_wk",
        "calendar_month",
        "calendar_year",

        "pe_article",
        "pe_store_group",
        "pe_article_store_group",

        "pe_quantity",
        "days_in_week",
        "days_sold_count",

        "pe_unit_price",
        "pe_actual_retail_price",
        "pe_average_zone_retail_price",
        "discount",
        "valid_price_flag",
        "price_available_flag",
        "price_data_quality_flag",
        "price_start_date",
        "price_end_date",

        "pe_sales_amount",

        "pe_store_stock_quantity",
        "pe_inventory_onhand_quantity",
        "pe_store_stock_quantity_for_model",
        "pe_inventory_onhand_quantity_for_model",
        "stock_available_flag",
        "inventory_available_flag",

        "is_sold",
        "probability_target",

        "valid_for_price_elasticity",
        "valid_for_mdo_input",
        "missing_price_reason",
        "missing_stock_reason",
        "missing_inventory_reason",

        "pe_gender",
        "pe_category",
        "pe_product_type",
        "pe_product_division",

        "pe_store_state",
        "pe_store_type",
        "pe_country",

        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "snap",

        "sellout_product_group",

        "article_row_surrogate_id",
        "product_row_surrogate_id",
        "store_row_surrogate_id",

        "sellout_row_hash_id",
        "price_row_hash_id",
        "stock_row_hash_id",
        "inventory_row_hash_id",

        "sellout_batch_id",
        "price_batch_id",
        "stock_batch_id",
        "inventory_batch_id",
        "article_batch_id",
        "product_batch_id",
        "store_batch_id",

        "sellout_project_id",
        "price_project_id",
        "stock_project_id",
        "inventory_project_id",
        "article_project_id",
        "product_project_id",
        "store_project_id",
    ]

    available_columns = [
        col_name
        for col_name in selected_columns
        if col_name in base_df.columns
    ]

    base_df = base_df.select(*available_columns)

    print("Final base rows:", base_df.count())
    print("Final base columns:", len(base_df.columns))

    return base_df


# ============================================================
# 9. Quality summary
# ============================================================

def create_quality_summary(base_df):
    quality_df = (
        base_df
        .groupBy("pe_store_group")
        .agg(
            F.count("*").alias("rows"),
            F.countDistinct("pe_article").alias("products"),
            F.countDistinct("wm_yr_wk").alias("weeks"),
            F.countDistinct("pe_article_store_group").alias("product_store_groups"),

            F.sum("price_available_flag").alias("rows_with_price"),
            F.sum("valid_price_flag").alias("rows_with_valid_price"),
            F.sum("stock_available_flag").alias("rows_with_stock"),
            F.sum("inventory_available_flag").alias("rows_with_inventory"),
            F.sum("valid_for_price_elasticity").alias("rows_valid_for_price_elasticity"),
            F.sum("valid_for_mdo_input").alias("rows_valid_for_mdo_input"),

            F.min("date").alias("min_date"),
            F.max("date").alias("max_date"),
        )
        .withColumn("price_coverage_pct", F.col("rows_with_price") / F.col("rows"))
        .withColumn("stock_coverage_pct", F.col("rows_with_stock") / F.col("rows"))
        .withColumn("inventory_coverage_pct", F.col("rows_with_inventory") / F.col("rows"))
        .withColumn("mdo_ready_pct", F.col("rows_valid_for_mdo_input") / F.col("rows"))
    )

    return quality_df


# ============================================================
# 10. Save and validate
# ============================================================

def run_base_data_build():
    print("Dropping existing output tables if present.")
    spark.sql(f"DROP TABLE IF EXISTS {OUTPUT_BASE_TABLE}")
    spark.sql(f"DROP TABLE IF EXISTS {OUTPUT_QUALITY_TABLE}")

    base_df = create_base_data_table()

    print("Saving base table:", OUTPUT_BASE_TABLE)

    (
        base_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(OUTPUT_BASE_TABLE)
    )

    quality_df = create_quality_summary(base_df)

    print("Saving quality summary table:", OUTPUT_QUALITY_TABLE)

    (
        quality_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(OUTPUT_QUALITY_TABLE)
    )

    print("Validation 1: table size")
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
            FROM {OUTPUT_BASE_TABLE}
        """)
    )

    print("Validation 2: store-level quality")
    display(
        spark.sql(f"""
            SELECT *
            FROM {OUTPUT_QUALITY_TABLE}
            ORDER BY pe_store_group
        """)
    )

    print("Validation 3: rows per product-store")
    display(
        spark.sql(f"""
            SELECT
                MIN(rows_per_group) AS min_rows_per_group,
                MAX(rows_per_group) AS max_rows_per_group,
                AVG(rows_per_group) AS avg_rows_per_group
            FROM (
                SELECT
                    pe_article_store_group,
                    COUNT(*) AS rows_per_group
                FROM {OUTPUT_BASE_TABLE}
                GROUP BY pe_article_store_group
            )
        """)
    )

    print("Validation 4: price and discount movement for valid price rows")
    display(
        spark.sql(f"""
            SELECT
                MIN(price_variation_count) AS min_price_variation,
                MAX(price_variation_count) AS max_price_variation,
                AVG(price_variation_count) AS avg_price_variation,
                MIN(discount_variation_count) AS min_discount_variation,
                MAX(discount_variation_count) AS max_discount_variation,
                AVG(discount_variation_count) AS avg_discount_variation
            FROM (
                SELECT
                    pe_article_store_group,
                    COUNT(DISTINCT pe_unit_price) AS price_variation_count,
                    COUNT(DISTINCT discount) AS discount_variation_count
                FROM {OUTPUT_BASE_TABLE}
                WHERE valid_price_flag = 1
                GROUP BY pe_article_store_group
            )
        """)
    )

    print("Validation 5: sold and not-sold rows")
    display(
        spark.sql(f"""
            SELECT
                is_sold,
                COUNT(*) AS rows
            FROM {OUTPUT_BASE_TABLE}
            GROUP BY is_sold
            ORDER BY is_sold
        """)
    )

    print("Sample output")
    display(
        spark.sql(f"""
            SELECT
                date,
                week_start_date,
                wm_yr_wk,
                pe_article,
                pe_store_group,
                pe_article_store_group,
                pe_quantity,
                pe_unit_price,
                pe_actual_retail_price,
                pe_average_zone_retail_price,
                discount,
                valid_price_flag,
                price_available_flag,
                stock_available_flag,
                inventory_available_flag,
                valid_for_price_elasticity,
                valid_for_mdo_input,
                pe_sales_amount,
                pe_store_stock_quantity,
                pe_inventory_onhand_quantity,
                is_sold,
                probability_target,
                pe_category,
                pe_product_type,
                pe_product_division,
                pe_store_state,
                snap
            FROM {OUTPUT_BASE_TABLE}
            ORDER BY pe_store_group, pe_article, wm_yr_wk
            LIMIT 100
        """)
    )

    print("====================================================")
    print("base_data_table build completed")
    print("Base output:", OUTPUT_BASE_TABLE)
    print("Quality output:", OUTPUT_QUALITY_TABLE)
    print("====================================================")

    return base_df, quality_df


# ============================================================
# 11. Execute
# ============================================================

base_df, quality_df = run_base_data_build()

if __name__ == "__main__":
    run_base_data_build()