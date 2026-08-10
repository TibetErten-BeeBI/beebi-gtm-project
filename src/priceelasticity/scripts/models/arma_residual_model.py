from typing import Optional

from pyspark.sql import functions as F
from pyspark.sql import Window


class ARMAResidualModel:
    """
    Daily-safe ARMA-style residual model.

    Old project idea:
        main model prediction
        + ARMA residual forecast
        = corrected prediction

    Daily version:
        1. Uses product-store group residual history.
        2. Orders residuals by exact date when date_col is provided.
        3. Estimates simple AR(1)-style residual behavior.
        4. Forecasts one residual correction per product-store group.
        5. Caps correction for business/model safety.
        6. Lets predictive_model_dev.py decide whether to use it based on validation RMSE.

    Expected daily input grain:
        pe_article_store_group + date

    Compatibility:
        - New predictive code can pass date_col.
        - Old predictive code can still pass week_col.
    """

    def __init__(
        self,
        group_col: str,
        week_col: Optional[str] = None,
        residual_col: Optional[str] = None,
        date_col: Optional[str] = None,
        time_col: Optional[str] = None,
        min_history_points: int = 28,
        max_abs_correction: float = 0.75,
        phi_floor: float = -0.80,
        phi_ceiling: float = 0.80,
    ):
        self.group_col = group_col

        # Daily preferred order:
        # 1. date_col
        # 2. time_col
        # 3. week_col, for backward compatibility
        self.date_col = date_col
        self.time_col = time_col
        self.week_col = week_col

        self.order_col = date_col or time_col or week_col

        if residual_col is None:
            raise ValueError("residual_col is required for ARMAResidualModel.")

        if self.order_col is None:
            raise ValueError(
                "ARMAResidualModel requires one time column. "
                "Pass date_col for daily data, or week_col for old weekly data."
            )

        self.residual_col = residual_col

        # Daily default:
        # 28 points means roughly 4 weeks of daily history.
        self.min_history_points = int(min_history_points)
        self.max_abs_correction = float(max_abs_correction)
        self.phi_floor = float(phi_floor)
        self.phi_ceiling = float(phi_ceiling)

        self.model_df = None

    def fit(self, residual_df):
        """
        Train ARMA-style residual model.

        Daily input residual_df should contain:
            group_col
            date_col
            residual_col

        Weekly compatibility input can contain:
            group_col
            week_col
            residual_col

        Output stored in self.model_df:
            group_col
            arma_history_points
            arma_mean_residual
            arma_last_residual
            arma_phi_raw
            arma_phi
            arma_residual_correction_raw
            arma_residual_correction
            arma_last_date
            arma_last_week
        """

        required_cols = [
            self.group_col,
            self.order_col,
            self.residual_col,
        ]

        missing_cols = [
            col_name
            for col_name in required_cols
            if col_name not in residual_df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing required columns for ARMAResidualModel.fit: {missing_cols}"
            )

        clean_df = (
            residual_df
            .select(
                F.col(self.group_col).alias(self.group_col),
                F.col(self.order_col).alias("_arma_time_raw"),
                F.col(self.residual_col).cast("double").alias(self.residual_col),
            )
            .withColumn("_arma_time_string", F.col("_arma_time_raw").cast("string"))
            .withColumn("_arma_time_date", F.to_date(F.col("_arma_time_string")))
            .withColumn(
                "_arma_time_numeric",
                F.expr("try_cast(_arma_time_string as BIGINT)")
            )
            .withColumn(
                "_arma_time_order",
                F.coalesce(
                    F.datediff(
                        F.col("_arma_time_date"),
                        F.lit("1970-01-01").cast("date")
                    ).cast("long"),
                    F.col("_arma_time_numeric").cast("long")
                )
            )
            .filter(F.col(self.group_col).isNotNull())
            .filter(F.col("_arma_time_order").isNotNull())
            .filter(F.col(self.residual_col).isNotNull())
        )

        # If duplicates exist for same product-store-date, average residual.
        # This protects the model, but it does not change correctly unique daily data.
        clean_df = (
            clean_df
            .groupBy(
                self.group_col,
                "_arma_time_order",
                "_arma_time_date",
                "_arma_time_string",
            )
            .agg(
                F.avg(F.col(self.residual_col)).alias(self.residual_col)
            )
        )

        group_window = Window.partitionBy(self.group_col).orderBy("_arma_time_order")

        last_window = (
            Window
            .partitionBy(self.group_col)
            .orderBy(F.col("_arma_time_order").desc())
        )

        lagged_df = (
            clean_df
            .withColumn(
                "_lag_residual",
                F.lag(F.col(self.residual_col)).over(group_window)
            )
            .withColumn(
                "_row_desc",
                F.row_number().over(last_window)
            )
        )

        group_stats_df = (
            lagged_df
            .groupBy(self.group_col)
            .agg(
                F.count("*").alias("arma_history_points"),
                F.avg(F.col(self.residual_col)).alias("arma_mean_residual"),
                F.sum(
                    F.when(
                        F.col("_lag_residual").isNotNull(),
                        F.col(self.residual_col) * F.col("_lag_residual")
                    ).otherwise(F.lit(0.0))
                ).alias("_phi_numerator"),
                F.sum(
                    F.when(
                        F.col("_lag_residual").isNotNull(),
                        F.col("_lag_residual") * F.col("_lag_residual")
                    ).otherwise(F.lit(0.0))
                ).alias("_phi_denominator"),
                F.sum(
                    F.when(
                        F.col("_lag_residual").isNotNull(),
                        F.lit(1)
                    ).otherwise(F.lit(0))
                ).alias("_lag_pairs"),
            )
        )

        last_residual_df = (
            lagged_df
            .filter(F.col("_row_desc") == 1)
            .select(
                self.group_col,
                F.col(self.residual_col).alias("arma_last_residual"),
                F.col("_arma_time_order").alias("arma_last_time_order"),
                F.col("_arma_time_date").alias("arma_last_date"),

                # Compatibility column.
                # For daily model this will contain date string.
                # For old weekly model this will contain week value as string.
                F.col("_arma_time_string").alias("arma_last_week"),
            )
        )

        model_df = (
            group_stats_df
            .join(last_residual_df, on=self.group_col, how="left")
            .withColumn(
                "arma_phi_raw",
                F.when(
                    (F.col("_lag_pairs") >= F.lit(2)) &
                    (F.abs(F.col("_phi_denominator")) > F.lit(1e-9)),
                    F.col("_phi_numerator") / F.col("_phi_denominator")
                ).otherwise(F.lit(0.0))
            )
            .withColumn(
                "arma_phi",
                F.least(
                    F.greatest(
                        F.col("arma_phi_raw"),
                        F.lit(self.phi_floor)
                    ),
                    F.lit(self.phi_ceiling)
                )
            )
            .withColumn(
                "arma_residual_correction_raw",
                F.when(
                    F.col("arma_history_points") >= F.lit(self.min_history_points),
                    F.col("arma_mean_residual")
                    + F.col("arma_phi") * (
                        F.col("arma_last_residual") - F.col("arma_mean_residual")
                    )
                ).otherwise(F.lit(0.0))
            )
            .withColumn(
                "arma_residual_correction",
                F.least(
                    F.greatest(
                        F.col("arma_residual_correction_raw"),
                        F.lit(-self.max_abs_correction)
                    ),
                    F.lit(self.max_abs_correction)
                )
            )
            .withColumn(
                "arma_min_history_points",
                F.lit(self.min_history_points)
            )
            .withColumn(
                "arma_max_abs_correction",
                F.lit(self.max_abs_correction)
            )
            .withColumn(
                "arma_time_column_used",
                F.lit(self.order_col)
            )
            .withColumn(
                "arma_grain_assumption",
                F.lit("product_store_day" if self.date_col else "product_store_time")
            )
            .drop("_phi_numerator", "_phi_denominator", "_lag_pairs")
        )

        self.model_df = model_df

        return self

    def transform(
        self,
        score_df,
        output_col: str = "arma_residual_correction",
    ):
        """
        Add ARMA residual correction columns to scoring data.

        If a product-store group was not trained, correction is zero.

        Daily behavior:
            Correction is learned from the latest daily residual pattern
            for each product-store group.
        """

        if self.model_df is None:
            raise ValueError("ARMAResidualModel must be fit before transform.")

        if self.group_col not in score_df.columns:
            raise ValueError(f"Missing group column in scoring data: {self.group_col}")

        scored_df = (
            score_df
            .join(
                self.model_df,
                on=self.group_col,
                how="left"
            )
            .withColumn(
                "arma_history_points",
                F.coalesce(F.col("arma_history_points"), F.lit(0))
            )
            .withColumn(
                "arma_mean_residual",
                F.coalesce(F.col("arma_mean_residual"), F.lit(0.0))
            )
            .withColumn(
                "arma_last_residual",
                F.coalesce(F.col("arma_last_residual"), F.lit(0.0))
            )
            .withColumn(
                "arma_phi_raw",
                F.coalesce(F.col("arma_phi_raw"), F.lit(0.0))
            )
            .withColumn(
                "arma_phi",
                F.coalesce(F.col("arma_phi"), F.lit(0.0))
            )
            .withColumn(
                "arma_residual_correction_raw",
                F.coalesce(F.col("arma_residual_correction_raw"), F.lit(0.0))
            )
            .withColumn(
                "arma_residual_correction",
                F.coalesce(F.col("arma_residual_correction"), F.lit(0.0))
            )
            .withColumn(
                output_col,
                F.coalesce(F.col("arma_residual_correction"), F.lit(0.0))
            )
            .withColumn(
                "arma_min_history_points",
                F.coalesce(
                    F.col("arma_min_history_points"),
                    F.lit(self.min_history_points)
                )
            )
            .withColumn(
                "arma_max_abs_correction",
                F.coalesce(
                    F.col("arma_max_abs_correction"),
                    F.lit(self.max_abs_correction)
                )
            )
            .withColumn(
                "arma_last_time_order",
                F.col("arma_last_time_order")
            )
            .withColumn(
                "arma_last_date",
                F.col("arma_last_date").cast("date")
            )
            .withColumn(
                "arma_last_week",
                F.col("arma_last_week").cast("string")
            )
            .withColumn(
                "arma_time_column_used",
                F.coalesce(
                    F.col("arma_time_column_used"),
                    F.lit(self.order_col)
                )
            )
            .withColumn(
                "arma_grain_assumption",
                F.coalesce(
                    F.col("arma_grain_assumption"),
                    F.lit("product_store_day" if self.date_col else "product_store_time")
                )
            )
        )

        return scored_df

    def get_model_df(self):
        """
        Return trained group-level ARMA residual table.
        """

        if self.model_df is None:
            raise ValueError("ARMAResidualModel has not been fit yet.")

        return self.model_df