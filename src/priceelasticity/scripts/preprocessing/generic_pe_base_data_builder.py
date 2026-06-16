from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import pandas as pd
import re
import sys


# ============================================================
# 1. Spark setup
# ============================================================

spark = SparkSession.builder.getOrCreate()

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
# ============================================================
# 2. Notebook parameters
# ============================================================
# In Databricks, you will set these values from widgets.
# You do NOT need to change code every time.
#
# Minimum required business fields:
#   product/article column
#   store/location column
#   date or week column
#   quantity/sales column
#   price column
#
# If your column names are common, auto-detect will work.
# If not, pass the column names through widgets.

INPUT_PATH = get_param("input_path", "")
INPUT_FORMAT = get_param("input_format", "csv")
EXCEL_SHEET = get_param("excel_sheet", "")
OUTPUT_SCHEMA = get_param("output_schema", "workspace.default")
TABLE_PREFIX = get_param("table_prefix", "")

MANUAL_MAP = {
    "product": get_param("product_col", ""),
    "store": get_param("store_col", ""),
    "date": get_param("date_col", ""),
    "week": get_param("week_col", ""),
    "quantity": get_param("quantity_col", ""),
    "price": get_param("price_col", ""),
    "base_price": get_param("base_price_col", ""),
    "discount": get_param("discount_col", ""),
    "country": get_param("country_col", ""),
    "category": get_param("category_col", ""),
    "product_type": get_param("product_type_col", ""),
    "product_division": get_param("product_division_col", ""),
    "gender": get_param("gender_col", ""),
    "stock": get_param("stock_col", ""),
    "inventory": get_param("inventory_col", ""),
}

if TABLE_PREFIX:
    OUTPUT_BASE_TABLE = f"{OUTPUT_SCHEMA}.{TABLE_PREFIX}_base_data_table"
    OUTPUT_QUALITY_TABLE = f"{OUTPUT_SCHEMA}.{TABLE_PREFIX}_base_data_quality_summary"
else:
    OUTPUT_BASE_TABLE = f"{OUTPUT_SCHEMA}.base_data_table"
    OUTPUT_QUALITY_TABLE = f"{OUTPUT_SCHEMA}.base_data_quality_summary"


# ============================================================
# 3. Auto-detection candidates
# ============================================================

COLUMN_CANDIDATES = {
    "product": [
        "pe_article",
        "group_article_id",
        "group_article",
        "article_id",
        "item_id",
        "product_id",
        "sku",
        "sku_id",
        "material",
        "material_id",
    ],
    "store": [
        "pe_store_group",
        "store_pos",
        "store",
        "store_id",
        "store_number",
        "reporting_unit",
        "reporting_unit_group",
        "location",
        "location_id",
    ],
    "date": [
        "date",
        "calendar_date",
        "sales_date",
        "transaction_date",
        "invoice_date",
        "snapshot_date",
        "value_date",
        "posting_date",
        "week_start_date",
    ],
    "week": [
        "wm_yr_wk",
        "week",
        "week_id",
        "week_number",
        "iso_week",
        "year_week",
        "week_start_date",
    ],
    "quantity": [
        "pe_quantity",
        "sold_quantity",
        "sold_qty",
        "quantity_sold",
        "sales_quantity",
        "net_quantity",
        "net_qty",
        "demand_qty",
        "units_sold",
        "qty",
        "quantity",
    ],
    "price": [
        "pe_unit_price",
        "actual_retail_price",
        "unit_price",
        "unit_price_eur",
        "selling_price",
        "sales_price",
        "price",
        "arp",
    ],
    "base_price": [
        "pe_average_zone_retail_price",
        "average_zone_retail_price",
        "planned_rrp",
        "rrp",
        "list_price",
        "regular_price",
        "base_price",
        "azrp",
    ],
    "discount": [
        "discount",
        "discount_pct",
        "discount_percent",
        "markdown_pct",
        "promo_discount",
        "scenario_discount",
    ],
    "country": [
        "pe_country",
        "country",
        "country_name",
        "market",
        "region",
    ],
    "category": [
        "pe_category",
        "category",
        "category_descr",
        "key_category_descr",
        "merchandise_category_description",
    ],
    "product_type": [
        "pe_product_type",
        "product_type",
        "product_type_descr",
        "merchandise_product_type_description",
    ],
    "product_division": [
        "pe_product_division",
        "division",
        "product_division",
        "product_division_descr",
        "merchandise_product_division_description",
    ],
    "gender": [
        "pe_gender",
        "gender",
        "gender_descr",
        "merchandise_gender_description",
    ],
    "stock": [
        "pe_store_stock_quantity",
        "stock",
        "stock_qty",
        "store_stock_quantity",
        "store_stock_quantity_base_unit_of_measure",
        "on_hand_stock_quantity",
    ],
    "inventory": [
        "pe_inventory_onhand_quantity",
        "inventory",
        "inventory_qty",
        "onhandstockqty",
        "on_hand_stock_quantity",
    ],
}


# ============================================================
# 4. Helpers
# ============================================================

def normalize_col_name(col_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col_name).strip().lower()).strip("_")


def normalize_dataframe_columns(df):
    new_cols = []
    seen = {}

    for col_name in df.columns:
        clean_name = normalize_col_name(col_name)

        if clean_name in seen:
            seen[clean_name] += 1
            clean_name = f"{clean_name}_{seen[clean_name]}"
        else:
            seen[clean_name] = 0

        new_cols.append(clean_name)

    return df.toDF(*new_cols)


def dbfs_to_local_path(path: str) -> str:
    if path.startswith("dbfs:/"):
        return "/dbfs/" + path.replace("dbfs:/", "", 1)
    return path


def clean_manual_map(manual_map):
    cleaned = {}

    for key, value in manual_map.items():
        value = str(value).strip()

        if value == "":
            cleaned[key] = ""
        else:
            cleaned[key] = normalize_col_name(value)

    return cleaned


def find_column(df, role: str, required: bool = False):
    manual_value = CLEAN_MANUAL_MAP.get(role, "")

    if manual_value and manual_value in df.columns:
        return manual_value

    for candidate in COLUMN_CANDIDATES.get(role, []):
        candidate_clean = normalize_col_name(candidate)

        if candidate_clean in df.columns:
            return candidate_clean

    if required:
        raise ValueError(
            f"Missing required column for role '{role}'. "
            f"Pass it using widget '{role}_col'. "
            f"Available columns: {df.columns}"
        )

    return None


def safe_col(df, col_name, alias_name, cast_type=None, default_value=None):
    if col_name and col_name in df.columns:
        expr = F.col(col_name)
    else:
        expr = F.lit(default_value)

    if cast_type:
        expr = expr.cast(cast_type)

    return expr.alias(alias_name)


def load_input_dataset():
    if not INPUT_PATH:
        raise ValueError("input_path is empty. Set the dataset path first.")

    fmt = INPUT_FORMAT.lower().strip()

    if fmt == "csv":
        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(INPUT_PATH)
        )

    elif fmt == "parquet":
        df = spark.read.parquet(INPUT_PATH)

    elif fmt == "delta":
        # INPUT_PATH can be either a path or a table name.
        if INPUT_PATH.startswith("dbfs:/") or INPUT_PATH.startswith("/") or INPUT_PATH.startswith("s3:/") or INPUT_PATH.startswith("abfss:/"):
            df = spark.read.format("delta").load(INPUT_PATH)
        else:
            df = spark.table(INPUT_PATH)

    elif fmt == "excel":
        local_path = dbfs_to_local_path(INPUT_PATH)

        if EXCEL_SHEET.strip():
            pdf = pd.read_excel(local_path, sheet_name=EXCEL_SHEET.strip())
        else:
            pdf = pd.read_excel(local_path)

        df = spark.createDataFrame(pdf)

    else:
        raise ValueError(f"Unsupported input_format: {INPUT_FORMAT}")

    df = normalize_dataframe_columns(df)

    print("Input rows:", df.count())
    print("Input columns:", df.columns)

    return df


def add_week_fields(df, date_col, week_col):
    if date_col:
        df = df.withColumn("_raw_date", F.to_date(F.col(date_col)))
    else:
        df = df.withColumn("_raw_date", F.lit(None).cast("date"))

    if week_col and week_col in df.columns and week_col != date_col:
        # If week column is numeric like 202605, use it.
        # If it is a date, convert to year*100 + weekofyear.
        df = df.withColumn(
            "_raw_week_string",
            F.col(week_col).cast("string")
        )

        df = df.withColumn(
            "_week_as_date",
            F.to_date(F.col(week_col))
        )

        df = df.withColumn(
            "_wm_yr_wk_from_week",
            F.when(
                F.col("_raw_week_string").rlike(r"^[0-9]{5,6}$"),
                F.col(week_col).cast("int")
            ).when(
                F.col("_week_as_date").isNotNull(),
                (F.year(F.col("_week_as_date")) * F.lit(100)) + F.weekofyear(F.col("_week_as_date"))
            ).otherwise(F.lit(None).cast("int"))
        )
    else:
        df = df.withColumn("_wm_yr_wk_from_week", F.lit(None).cast("int"))
        df = df.withColumn("_week_as_date", F.lit(None).cast("date"))

    df = df.withColumn(
        "week_start_date",
        F.coalesce(
            F.col("_week_as_date"),
            F.to_date(F.date_trunc("week", F.col("_raw_date")))
        )
    )

    df = df.withColumn(
        "date",
        F.coalesce(
            F.col("_raw_date"),
            F.date_add(F.col("week_start_date"), 6)
        )
    )

    df = df.withColumn(
        "wm_yr_wk",
        F.coalesce(
            F.col("_wm_yr_wk_from_week"),
            (F.year(F.col("week_start_date")) * F.lit(100)) + F.weekofyear(F.col("week_start_date"))
        ).cast("int")
    )

    return df


# ============================================================
# 5. Main conversion
# ============================================================

def create_generic_base_data_table():
    print("==================================================")
    print("Running generic dataset to PE/MDO base table")
    print("==================================================")

    raw_df = load_input_dataset()

    product_col = find_column(raw_df, "product", required=True)
    store_col = find_column(raw_df, "store", required=True)
    date_col = find_column(raw_df, "date", required=False)
    week_col = find_column(raw_df, "week", required=False)
    quantity_col = find_column(raw_df, "quantity", required=True)
    price_col = find_column(raw_df, "price", required=True)

    base_price_col = find_column(raw_df, "base_price", required=False)
    discount_col = find_column(raw_df, "discount", required=False)

    country_col = find_column(raw_df, "country", required=False)
    category_col = find_column(raw_df, "category", required=False)
    product_type_col = find_column(raw_df, "product_type", required=False)
    product_division_col = find_column(raw_df, "product_division", required=False)
    gender_col = find_column(raw_df, "gender", required=False)

    stock_col = find_column(raw_df, "stock", required=False)
    inventory_col = find_column(raw_df, "inventory", required=False)

    if not date_col and not week_col:
        raise ValueError(
            "Dataset must contain either date column or week column. "
            "Set date_col or week_col widget."
        )

    print("Detected/mapped columns:")
    print("product:", product_col)
    print("store:", store_col)
    print("date:", date_col)
    print("week:", week_col)
    print("quantity:", quantity_col)
    print("price:", price_col)
    print("base_price:", base_price_col)
    print("discount:", discount_col)
    print("country:", country_col)
    print("category:", category_col)
    print("product_type:", product_type_col)
    print("product_division:", product_division_col)
    print("gender:", gender_col)
    print("stock:", stock_col)
    print("inventory:", inventory_col)

    df = add_week_fields(raw_df, date_col=date_col, week_col=week_col)

    standard_df = (
        df
        .select(
            F.col(product_col).cast("string").alias("pe_article"),
            F.col(store_col).cast("string").alias("pe_store_group"),

            F.col("date").alias("date"),
            F.col("week_start_date").alias("week_start_date"),
            F.col("wm_yr_wk").alias("wm_yr_wk"),

            F.col(quantity_col).cast("double").alias("raw_quantity"),
            F.col(price_col).cast("double").alias("raw_unit_price"),

            safe_col(df, base_price_col, "raw_base_price", "double"),
            safe_col(df, discount_col, "raw_discount", "double"),

            safe_col(df, country_col, "pe_country", "string", "UNKNOWN"),
            safe_col(df, category_col, "pe_category", "string", "UNKNOWN"),
            safe_col(df, product_type_col, "pe_product_type", "string", "UNKNOWN"),
            safe_col(df, product_division_col, "pe_product_division", "string", "UNKNOWN"),
            safe_col(df, gender_col, "pe_gender", "string", "UNKNOWN"),

            safe_col(df, stock_col, "raw_stock_quantity", "double"),
            safe_col(df, inventory_col, "raw_inventory_quantity", "double"),
        )
        .filter(F.col("pe_article").isNotNull())
        .filter(F.col("pe_store_group").isNotNull())
        .filter(F.col("date").isNotNull())
        .filter(F.col("wm_yr_wk").isNotNull())
        .filter(F.col("raw_quantity").isNotNull())
    )

    # Aggregate to product-store-week grain.
    base_df = (
        standard_df
        .groupBy(
            "pe_article",
            "pe_store_group",
            "wm_yr_wk"
        )
        .agg(
            F.min("week_start_date").alias("week_start_date"),
            F.max("date").alias("date"),

            F.sum(F.coalesce(F.col("raw_quantity"), F.lit(0.0))).alias("pe_quantity"),

            F.avg("raw_unit_price").alias("pe_actual_retail_price"),
            F.avg("raw_base_price").alias("pe_average_zone_retail_price"),
            F.avg("raw_discount").alias("input_discount"),

            F.avg("raw_stock_quantity").alias("pe_store_stock_quantity"),
            F.avg("raw_inventory_quantity").alias("pe_inventory_onhand_quantity"),

            F.first("pe_country", ignorenulls=True).alias("pe_country"),
            F.first("pe_category", ignorenulls=True).alias("pe_category"),
            F.first("pe_product_type", ignorenulls=True).alias("pe_product_type"),
            F.first("pe_product_division", ignorenulls=True).alias("pe_product_division"),
            F.first("pe_gender", ignorenulls=True).alias("pe_gender"),

            F.count("*").alias("days_in_week"),
            F.sum(
                F.when(F.col("raw_quantity") > 0, F.lit(1)).otherwise(F.lit(0))
            ).alias("days_sold_count"),
        )
    )

    base_df = (
        base_df
        .withColumn("calendar_month", F.month(F.col("date")))
        .withColumn("calendar_year", F.year(F.col("date")))

        .withColumn(
            "pe_unit_price",
            F.coalesce(
                F.col("pe_actual_retail_price"),
                F.col("pe_average_zone_retail_price")
            )
        )

        .withColumn(
            "price_available_flag",
            F.when(F.col("pe_unit_price").isNotNull(), F.lit(1)).otherwise(F.lit(0))
        )

        .withColumn(
            "valid_price_flag",
            F.when(F.col("pe_unit_price") > 0, F.lit(1)).otherwise(F.lit(0))
        )

        .withColumn(
            "discount_from_price",
            F.when(
                (F.col("pe_average_zone_retail_price") > 0) &
                (F.col("pe_actual_retail_price").isNotNull()) &
                (F.col("pe_actual_retail_price") < F.col("pe_average_zone_retail_price")),
                (
                    F.col("pe_average_zone_retail_price") -
                    F.col("pe_actual_retail_price")
                ) / F.col("pe_average_zone_retail_price")
            ).otherwise(F.lit(None).cast("double"))
        )

        .withColumn(
            "discount",
            F.coalesce(
                F.col("input_discount"),
                F.col("discount_from_price"),
                F.lit(0.0)
            )
        )

        # Handle discount given as percentage like 15 instead of 0.15.
        .withColumn(
            "discount",
            F.when(F.col("discount") > 1.0, F.col("discount") / F.lit(100.0))
             .otherwise(F.col("discount"))
        )

        .withColumn(
            "discount",
            F.when(F.col("discount") < 0, F.lit(0.0))
             .when(F.col("discount") > 0.80, F.lit(0.80))
             .otherwise(F.col("discount"))
        )

        .withColumn(
            "pe_sales_amount",
            F.when(
                F.col("pe_unit_price").isNotNull(),
                F.col("pe_quantity") * F.col("pe_unit_price")
            ).otherwise(F.lit(None).cast("double"))
        )

        .withColumn(
            "is_sold",
            F.when(F.col("pe_quantity") > 0, F.lit(1)).otherwise(F.lit(0))
        )
        .withColumn("probability_target", F.col("is_sold"))

        .withColumn(
            "stock_available_flag",
            F.when(F.col("pe_store_stock_quantity").isNotNull(), F.lit(1)).otherwise(F.lit(0))
        )
        .withColumn(
            "inventory_available_flag",
            F.when(F.col("pe_inventory_onhand_quantity").isNotNull(), F.lit(1)).otherwise(F.lit(0))
        )

        .withColumn(
            "pe_store_stock_quantity_for_model",
            F.coalesce(F.col("pe_store_stock_quantity"), F.lit(0.0))
        )
        .withColumn(
            "pe_inventory_onhand_quantity_for_model",
            F.coalesce(F.col("pe_inventory_onhand_quantity"), F.lit(0.0))
        )

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
             .otherwise(F.lit(None).cast("string"))
        )
        .withColumn(
            "missing_stock_reason",
            F.when(F.col("stock_available_flag") == 0, F.lit("missing_stock_source"))
             .otherwise(F.lit(None).cast("string"))
        )
        .withColumn(
            "missing_inventory_reason",
            F.when(F.col("inventory_available_flag") == 0, F.lit("missing_inventory_source"))
             .otherwise(F.lit(None).cast("string"))
        )

        .withColumn(
            "pe_article_store_group",
            F.concat_ws("_", F.col("pe_article"), F.col("pe_store_group"))
        )

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

        .withColumn("base_data_grain", F.lit("product_store_week"))

        .withColumn(
            "pe_store_state",
            F.when(
                F.col("pe_store_group").contains("_"),
                F.split(F.col("pe_store_group"), "_").getItem(0)
            ).otherwise(F.col("pe_country"))
        )
        .withColumn("pe_store_type", F.col("pe_store_state"))

        .withColumn(
            "price_data_quality_flag",
            F.when(F.col("valid_price_flag") == 1, F.lit("MAPPED"))
             .otherwise(F.lit("MISSING_OR_INVALID"))
        )
        .withColumn("price_start_date", F.col("week_start_date"))
        .withColumn("price_end_date", F.col("date"))

        # Optional calendar/event fields expected by feature code.
        .withColumn("event_name_1", F.lit(None).cast("string"))
        .withColumn("event_type_1", F.lit(None).cast("string"))
        .withColumn("event_name_2", F.lit(None).cast("string"))
        .withColumn("event_type_2", F.lit(None).cast("string"))
        .withColumn("snap_CA", F.lit(0))
        .withColumn("snap_TX", F.lit(0))
        .withColumn("snap_WI", F.lit(0))
        .withColumn("snap", F.lit(0))
    )

    final_cols = [
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
    ]

    base_df = (
        base_df
        .select(*final_cols)
        .filter(F.col("date").isNotNull())
        .filter(F.col("wm_yr_wk").isNotNull())
        .filter(F.col("pe_article").isNotNull())
        .filter(F.col("pe_store_group").isNotNull())
        .filter(F.col("pe_quantity").isNotNull())
        .filter(F.col("pe_quantity") >= 0)
        .dropDuplicates(["pe_article", "pe_store_group", "wm_yr_wk"])
    )

    print("Dropping old output tables if present.")
    spark.sql(f"DROP TABLE IF EXISTS {OUTPUT_BASE_TABLE}")
    spark.sql(f"DROP TABLE IF EXISTS {OUTPUT_QUALITY_TABLE}")

    print("Saving:", OUTPUT_BASE_TABLE)

    (
        base_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(OUTPUT_BASE_TABLE)
    )

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

    print("Saving:", OUTPUT_QUALITY_TABLE)

    (
        quality_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(OUTPUT_QUALITY_TABLE)
    )

    print("Validation 1: base table summary")
    display(
        spark.sql(f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT pe_article) AS products,
                COUNT(DISTINCT pe_store_group) AS stores,
                COUNT(DISTINCT pe_article_store_group) AS product_store_groups,
                COUNT(DISTINCT wm_yr_wk) AS weeks,
                MIN(date) AS min_date,
                MAX(date) AS max_date,
                SUM(valid_price_flag) AS rows_with_valid_price,
                SUM(valid_for_price_elasticity) AS rows_valid_for_price_elasticity,
                SUM(valid_for_mdo_input) AS rows_valid_for_mdo_input
            FROM {OUTPUT_BASE_TABLE}
        """)
    )

    print("Validation 2: price and discount variation")
    display(
        spark.sql(f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT pe_unit_price) AS price_values,
                COUNT(DISTINCT discount) AS discount_values,
                MIN(pe_unit_price) AS min_price,
                MAX(pe_unit_price) AS max_price,
                MIN(discount) AS min_discount,
                MAX(discount) AS max_discount,
                AVG(discount) AS avg_discount
            FROM {OUTPUT_BASE_TABLE}
            WHERE valid_price_flag = 1
        """)
    )

    print("Validation 3: quality by store")
    display(
        spark.sql(f"""
            SELECT *
            FROM {OUTPUT_QUALITY_TABLE}
            ORDER BY pe_store_group
        """)
    )

    print("Validation 4: sample rows")
    display(
        spark.sql(f"""
            SELECT
                date,
                week_start_date,
                wm_yr_wk,
                pe_article,
                pe_store_group,
                pe_quantity,
                pe_unit_price,
                pe_actual_retail_price,
                pe_average_zone_retail_price,
                discount,
                valid_price_flag,
                valid_for_price_elasticity,
                valid_for_mdo_input,
                pe_category,
                pe_product_type,
                pe_product_division
            FROM {OUTPUT_BASE_TABLE}
            ORDER BY pe_store_group, pe_article, wm_yr_wk
            LIMIT 100
        """)
    )

    print("==================================================")
    print("Generic PE/MDO base table completed successfully")
    print("Output:", OUTPUT_BASE_TABLE)
    print("Quality:", OUTPUT_QUALITY_TABLE)
    print("==================================================")

    return OUTPUT_BASE_TABLE


# ============================================================
# 6. Runner
# ============================================================

CLEAN_MANUAL_MAP = clean_manual_map(MANUAL_MAP)

if __name__ == "__main__":
    create_generic_base_data_table()