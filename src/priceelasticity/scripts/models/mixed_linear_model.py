from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class MixedLinearParameters:
    fe_params: Dict[str, float]
    random_effects_by_group: Dict[str, Dict[str, float]]


@dataclass
class MixedLinearResults:
    parameters: MixedLinearParameters
    model_type: str
    train_results_df: pd.DataFrame
    metadata: Dict[str, Any]


class MixedLinear:
    """
    Daily-safe Mixed Linear style model.

    Purpose:
        This class gives causal_model_dev.py the MixedLinear interface it expects.

    Daily grain:
        One training row should represent:

            date + pe_article + pe_store_group

        The group_col should usually be:

            pe_article_store_group

    What it does:
        1. Fits global fixed effects using ridge regression.
        2. Estimates simple group-level residual corrections using group-specific
           random intercept and random-effect columns.
        3. Returns results.parameters.fe_params so causal_model_dev.py can read
           discount_power coefficients.
        4. Keeps date-level training results, so downstream debugging can verify
           daily behavior.

    Important:
        This model does not aggregate weekly.
        It keeps the exact daily date passed through date_col.
    """

    def __init__(
        self,
        exog,
        group_col,
        endog,
        random_effects=None,
        date_col=None,
        ridge_alpha: float = 1e-5,
        random_alpha: float = 1.0,
        min_group_rows_for_random_effect: int = 28,
    ):
        self.exog = pd.DataFrame(exog).copy()
        self.group_col = pd.Series(group_col).copy()
        self.endog = pd.Series(endog).copy()

        self.random_effects = (
            pd.DataFrame(random_effects).copy()
            if random_effects is not None
            else None
        )

        self.date_col = (
            pd.Series(date_col).copy()
            if date_col is not None
            else None
        )

        self.ridge_alpha = float(ridge_alpha)
        self.random_alpha = float(random_alpha)

        # Daily change:
        # Old weekly minimum was usually 4 rows.
        # For daily, 4 weeks roughly means 28 rows.
        self.min_group_rows_for_random_effect = int(min_group_rows_for_random_effect)

    # ========================================================
    # Internal helpers
    # ========================================================

    def _validate_input_lengths(self) -> None:
        n_rows = len(self.exog)

        if len(self.group_col) != n_rows:
            raise ValueError(
                f"group_col length mismatch. "
                f"Expected {n_rows}, got {len(self.group_col)}."
            )

        if len(self.endog) != n_rows:
            raise ValueError(
                f"endog length mismatch. "
                f"Expected {n_rows}, got {len(self.endog)}."
            )

        if self.date_col is not None and len(self.date_col) != n_rows:
            raise ValueError(
                f"date_col length mismatch. "
                f"Expected {n_rows}, got {len(self.date_col)}."
            )

        if self.random_effects is not None and len(self.random_effects) != n_rows:
            raise ValueError(
                f"random_effects length mismatch. "
                f"Expected {n_rows}, got {len(self.random_effects)}."
            )

    @staticmethod
    def _ridge_fit(
        x: np.ndarray,
        y: np.ndarray,
        alpha: float,
        penalize_intercept: bool = False,
    ) -> np.ndarray:
        penalty = np.eye(x.shape[1]) * float(alpha)

        if not penalize_intercept and penalty.shape[0] > 0:
            penalty[0, 0] = 0.0

        beta = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y

        return beta

    @staticmethod
    def _safe_numeric_series(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").astype(float)

    def _prepare_training_frame(self) -> pd.DataFrame:
        self._validate_input_lengths()

        df = self.exog.copy()

        # Group and target
        df["_group_col"] = self.group_col.astype(str).values
        df["_endog"] = self._safe_numeric_series(self.endog).values

        # Daily date
        if self.date_col is not None:
            df["_date_col"] = pd.to_datetime(self.date_col, errors="coerce").values
        else:
            df["_date_col"] = pd.NaT

        # Random-effect columns
        if self.random_effects is not None:
            for col_name in self.random_effects.columns:
                df[f"_re_{col_name}"] = self._safe_numeric_series(
                    self.random_effects[col_name]
                ).values

        # Fixed-effect columns
        for col_name in self.exog.columns:
            df[col_name] = self._safe_numeric_series(df[col_name])

        numeric_cols = list(self.exog.columns) + ["_endog"]

        if self.random_effects is not None:
            numeric_cols += [
                f"_re_{col_name}"
                for col_name in self.random_effects.columns
            ]

        df = df.replace([np.inf, -np.inf], np.nan)

        required_cols = numeric_cols + ["_group_col"]

        # If date_col is provided, keep only valid daily dates.
        if self.date_col is not None:
            required_cols.append("_date_col")

        df = df.dropna(subset=required_cols).reset_index(drop=True)

        # Daily-safe sort. This does not aggregate.
        # It only makes output/debugging stable.
        if "_date_col" in df.columns:
            df = (
                df
                .sort_values(["_group_col", "_date_col"])
                .reset_index(drop=True)
            )

        df["_row_id"] = np.arange(len(df))

        return df

    def _fit_fixed_effects(self, df: pd.DataFrame) -> Tuple[Dict[str, float], np.ndarray]:
        feature_cols = list(self.exog.columns)

        x = df[feature_cols].astype(float).to_numpy()
        y = df["_endog"].astype(float).to_numpy()

        x_aug = np.column_stack([
            np.ones(len(x)),
            x,
        ])

        beta = self._ridge_fit(
            x=x_aug,
            y=y,
            alpha=self.ridge_alpha,
            penalize_intercept=False,
        )

        fe_params = {
            "intercept": float(beta[0])
        }

        for idx, col_name in enumerate(feature_cols, start=1):
            fe_params[col_name] = float(beta[idx])

        fixed_prediction = x_aug @ beta

        return fe_params, fixed_prediction

    def _fit_random_effects(
        self,
        df: pd.DataFrame,
        residual: np.ndarray,
    ) -> Tuple[Dict[str, Dict[str, float]], np.ndarray]:
        random_effects_by_group = {}

        if self.random_effects is None:
            return random_effects_by_group, np.zeros(len(df), dtype=float)

        re_cols = [
            f"_re_{col_name}"
            for col_name in self.random_effects.columns
            if f"_re_{col_name}" in df.columns
        ]

        if not re_cols:
            return random_effects_by_group, np.zeros(len(df), dtype=float)

        residual_df = df[["_group_col", "_row_id"] + re_cols].copy()
        residual_df["_residual"] = residual

        random_prediction = np.zeros(len(df), dtype=float)

        for group_value, group_pdf in residual_df.groupby("_group_col"):
            group_name = str(group_value)

            if len(group_pdf) < self.min_group_rows_for_random_effect:
                random_effects_by_group[group_name] = {
                    "random_intercept": 0.0,
                    **{
                        col_name.replace("_re_", "random_"): 0.0
                        for col_name in re_cols
                    },
                    "group_rows_used": int(len(group_pdf)),
                    "random_effect_applied": 0,
                }
                continue

            z = group_pdf[re_cols].astype(float).to_numpy()

            z_aug = np.column_stack([
                np.ones(len(z)),
                z,
            ])

            r = group_pdf["_residual"].astype(float).to_numpy()

            gamma = self._ridge_fit(
                x=z_aug,
                y=r,
                alpha=self.random_alpha,
                penalize_intercept=True,
            )

            effect_dict = {
                "random_intercept": float(gamma[0])
            }

            for idx, col_name in enumerate(re_cols, start=1):
                effect_dict[col_name.replace("_re_", "random_")] = float(gamma[idx])

            effect_dict["group_rows_used"] = int(len(group_pdf))
            effect_dict["random_effect_applied"] = 1

            random_effects_by_group[group_name] = effect_dict

            row_ids = group_pdf["_row_id"].astype(int).to_numpy()
            random_prediction[row_ids] = z_aug @ gamma

        return random_effects_by_group, random_prediction

    def _create_group_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        group_counts = df.groupby("_group_col").size()

        metadata = {
            "n_rows": int(len(df)),
            "n_groups": int(df["_group_col"].nunique()),
            "n_features": int(len(self.exog.columns)),
            "has_random_effects": bool(self.random_effects is not None),
            "ridge_alpha": float(self.ridge_alpha),
            "random_alpha": float(self.random_alpha),
            "min_group_rows_for_random_effect": int(self.min_group_rows_for_random_effect),
            "min_rows_per_group": int(group_counts.min()) if len(group_counts) > 0 else 0,
            "max_rows_per_group": int(group_counts.max()) if len(group_counts) > 0 else 0,
            "avg_rows_per_group": float(group_counts.mean()) if len(group_counts) > 0 else 0.0,
            "groups_with_random_effect_possible": int(
                (group_counts >= self.min_group_rows_for_random_effect).sum()
            ) if len(group_counts) > 0 else 0,
            "grain_assumption": "product_store_day",
        }

        if "_date_col" in df.columns and df["_date_col"].notna().any():
            date_stats = (
                df
                .dropna(subset=["_date_col"])
                .groupby("_group_col")["_date_col"]
                .agg(["min", "max", "nunique"])
            )

            metadata.update({
                "min_date": str(df["_date_col"].min()),
                "max_date": str(df["_date_col"].max()),
                "min_days_per_group": int(date_stats["nunique"].min()) if len(date_stats) > 0 else 0,
                "max_days_per_group": int(date_stats["nunique"].max()) if len(date_stats) > 0 else 0,
                "avg_days_per_group": float(date_stats["nunique"].mean()) if len(date_stats) > 0 else 0.0,
            })

            duplicate_daily_keys = (
                df
                .groupby(["_group_col", "_date_col"])
                .size()
                .reset_index(name="rows_per_group_date")
            )

            metadata["duplicate_group_date_keys"] = int(
                (duplicate_daily_keys["rows_per_group_date"] > 1).sum()
            )
        else:
            metadata.update({
                "min_date": None,
                "max_date": None,
                "min_days_per_group": 0,
                "max_days_per_group": 0,
                "avg_days_per_group": 0.0,
                "duplicate_group_date_keys": 0,
            })

        return metadata

    # ========================================================
    # Public training method
    # ========================================================

    def train(self, est_re_cov: bool = False) -> MixedLinearResults:
        df = self._prepare_training_frame()

        if df.empty:
            raise ValueError("MixedLinear cannot train because cleaned training data is empty.")

        fe_params, fixed_prediction = self._fit_fixed_effects(df)

        residual = df["_endog"].astype(float).to_numpy() - fixed_prediction

        random_effects_by_group, random_prediction = self._fit_random_effects(
            df=df,
            residual=residual,
        )

        final_prediction = fixed_prediction + random_prediction
        final_residual = df["_endog"].astype(float).to_numpy() - final_prediction

        train_results_df = pd.DataFrame({
            "date": df["_date_col"],
            "group_col": df["_group_col"],
            "actual": df["_endog"].astype(float),
            "prediction_fixed": fixed_prediction,
            "prediction_random": random_prediction,
            "prediction": final_prediction,
            "residual": final_residual,
        })

        metadata = self._create_group_metadata(df)
        metadata["est_re_cov"] = bool(est_re_cov)

        return MixedLinearResults(
            parameters=MixedLinearParameters(
                fe_params=fe_params,
                random_effects_by_group=random_effects_by_group,
            ),
            model_type="mixed_linear",
            train_results_df=train_results_df,
            metadata=metadata,
        )