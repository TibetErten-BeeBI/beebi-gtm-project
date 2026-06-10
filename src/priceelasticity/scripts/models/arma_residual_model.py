from typing import Optional

from pyspark.sql import functions as F
from pyspark.sql import Window


class ARMAResidualModel:
    """
    Serverless-safe ARMA-style residual model.

    Old project ideology:
        main model prediction
        + ARMA residual forecast
        = corrected prediction

    New project implementation:
        1. Calculate residual pattern by product-store group.
        2. Estimate AR(1)-style residual behavior using lag residuals.
        3. Forecast one residual correction per product-store group.
        4. Cap correction for business/model safety.
        5. Let predictive_model_dev.py decide whether to use it based on validation RMSE.

    This is not copied from the old project.
    It follows the same modeling idea in a simpler Spark-safe way.
    """

    def __init__(
        self,
        group_col: str,
        week_col: str,
        residual_col: str,
        min_history_points: int = 8,
        max_abs_correction: float = 0.75,
        phi_floor: float = -0.80,
        phi_ceiling: float = 0.80,
    ):
        self.group_col = group_col
        self.week_col = week_col
        self.residual_col = residual_col
        self.min_history_points = int(min_history_points)
        self.max_abs_correction = float(max_abs_correction)
        self.phi_floor = float(phi_floor)
        self.phi_ceiling = float(phi_ceiling)

        self.model_df = None

    def fit(self, residual_df):
        """
        Train ARMA-style residual model.

        Input residual_df must contain:
            group_col
            week_col
            residual_col

        Output stored in self.model_df:
            group_col
            arma_history_points
            arma_mean_residual
            arma_last_residual
            arma_phi
            arma_residual_correction_raw
            arma_residual_correction
        """

        required_cols = [
            self.group_col,
            self.week_col,
            self.residual_col,
        ]

        missing_cols = [
            col_name
            for col_name in required_cols
            if col_name not in residual_df.columns
        ]

        if missing_cols:
            raise ValueError(f"Missing required columns for ARMAResidualModel.fit: {missing_cols}")

        clean_df = (
            residual_df
            .select(
                F.col(self.group_col).alias(self.group_col),
                F.col(self.week_col).cast("long").alias(self.week_col),
                F.col(self.residual_col).cast("double").alias(self.residual_col),
            )
            .filter(F.col(self.group_col).isNotNull())
            .filter(F.col(self.week_col).isNotNull())
            .filter(F.col(self.residual_col).isNotNull())
        )

        group_window = Window.partitionBy(self.group_col).orderBy(self.week_col)
        last_window = Window.partitionBy(self.group_col).orderBy(F.col(self.week_col).desc())

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
                F.col(self.week_col).alias("arma_last_week"),
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
                output_col,
                F.coalesce(F.col("arma_residual_correction"), F.lit(0.0))
            )
            .withColumn(
                "arma_min_history_points",
                F.coalesce(F.col("arma_min_history_points"), F.lit(self.min_history_points))
            )
            .withColumn(
                "arma_max_abs_correction",
                F.coalesce(F.col("arma_max_abs_correction"), F.lit(self.max_abs_correction))
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