from pyspark.sql import SparkSession
import pandas as pd
import sys

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


def get_dbutils():
    try:
        return dbutils
    except NameError:
        from pyspark.dbutils import DBUtils
        return DBUtils(spark)


dbutils_ref = get_dbutils()

input_path = get_param("input_path", "")
input_format = get_param("input_format", "csv").lower()
excel_sheet = get_param("excel_sheet", "")
output_schema = get_param("output_schema", "workspace.default")
table_prefix = get_param("table_prefix", "")

if not input_path:
    raise ValueError("input_path is required")

if not table_prefix:
    raise ValueError("table_prefix is required")

raw_delta_table = f"{output_schema}.{table_prefix}_raw_delta"

print("Input path:", input_path)
print("Input format:", input_format)
print("Output raw delta table:", raw_delta_table)


def volume_path_to_local_path(path: str) -> str:
    if path.startswith("/Volumes/"):
        return path
    if path.startswith("dbfs:/"):
        return "/dbfs/" + path.replace("dbfs:/", "")
    return path


if input_format == "csv":
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(input_path)
    )

elif input_format == "parquet":
    df = spark.read.parquet(input_path)

elif input_format == "delta":
    print("Input is already Delta. Skipping conversion.")
    dbutils_ref.jobs.taskValues.set(key="raw_delta_table", value=input_path)
    dbutils_ref.notebook.exit(input_path)

elif input_format in ["excel", "xlsx"]:
    local_path = volume_path_to_local_path(input_path)

    if excel_sheet:
        pdf = pd.read_excel(local_path, sheet_name=excel_sheet)
    else:
        pdf = pd.read_excel(local_path)

    df = spark.createDataFrame(pdf)

else:
    raise ValueError(f"Unsupported input_format: {input_format}")

row_count = df.count()
column_count = len(df.columns)

print("Raw file row count:", row_count)
print("Raw file column count:", column_count)
print("Raw file columns:", df.columns)

if row_count == 0:
    raise ValueError("Uploaded file has zero rows. Stopping pipeline.")

df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(raw_delta_table)

print(f"Created raw Delta table successfully: {raw_delta_table}")

dbutils_ref.jobs.taskValues.set(key="raw_delta_table", value=raw_delta_table)