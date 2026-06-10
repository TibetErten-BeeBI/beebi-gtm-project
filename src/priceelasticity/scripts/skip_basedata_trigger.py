from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

BASE_TABLE = "workspace.default.base_data_table"

print("Skipping base data build.")
print(f"Checking existing base table: {BASE_TABLE}")

row_count = spark.table(BASE_TABLE).limit(1).count()

print(f"Base table exists. Sample check row count: {row_count}")
print("Continuing pipeline.")