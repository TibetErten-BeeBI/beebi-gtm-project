import math
from dataclasses import dataclass
from typing import Dict, Optional, Any

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
    Serverless-safe Mixed Linear style model.

    Purpose:
        This class gives causal_model_dev.py the MixedLinear interface it expects.

    What it does:
        1. Fits global fixed effects using ridge regression.
        2. Estimates simple group-level residual corrections using group-specific
           random intercept and random-effect columns.
        3. Returns results.parameters.fe_params so causal_model_dev.py can read
           discount_power coefficients.

    This avoids copying the old project implementation directly.
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
        min_group_rows_for_random_effect: int = 4,
    ):
        self.exog = pd.DataFrame(exog).copy()
        self.group_col = pd.Series(group_col).copy()
        self.endog = pd.Series(endog).copy()
        self.random_effects = (
            pd.DataFrame(random_effects).copy()
            if random_effects is not None
            else None
        )
        self.date_col = pd.Series(date_col).copy() if date_col is not None else None

        self.ridge_alpha = float(ridge_alpha)
        self.random_alpha = float(random_alpha)
        self.min_group_rows_for_random_effect = int(min_group_rows_for_random_effect)

    def _prepare_training_frame(self) -> pd.DataFrame:
        df = self.exog.copy()

        df["_group_col"] = self.group_col.astype(str).values
        df["_endog"] = pd.to_numeric(self.endog, errors="coerce").astype(float).values

        if self.date_col is not None:
            df["_date_col"] = pd.to_datetime(self.date_col, errors="coerce").values
        else:
            df["_date_col"] = pd.NaT

        if self.random_effects is not None:
            for col_name in self.random_effects.columns:
                df[f"_re_{col_name}"] = pd.to_numeric(
                    self.random_effects[col_name],
                    errors="coerce"
                ).astype(float).values

        for col_name in self.exog.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype(float)

        numeric_cols = list(self.exog.columns) + ["_endog"]

        if self.random_effects is not None:
            numeric_cols += [
                f"_re_{col_name}"
                for col_name in self.random_effects.columns
            ]

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=numeric_cols + ["_group_col"]).reset_index(drop=True)

        return df

    @staticmethod
    def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float, penalize_intercept: bool = False) -> np.ndarray:
        penalty = np.eye(x.shape[1]) * float(alpha)

        if not penalize_intercept and penalty.shape[0] > 0:
            penalty[0, 0] = 0.0

        beta = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y

        return beta

    def _fit_fixed_effects(self, df: pd.DataFrame):
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

    def _fit_random_effects(self, df: pd.DataFrame, residual: np.ndarray):
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

        residual_df = df[["_group_col"] + re_cols].copy()
        residual_df["_residual"] = residual
        residual_df["_row_id"] = np.arange(len(residual_df))

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

            random_effects_by_group[group_name] = effect_dict

            row_ids = group_pdf["_row_id"].astype(int).to_numpy()
            random_prediction[row_ids] = z_aug @ gamma

        return random_effects_by_group, random_prediction

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

        metadata = {
            "n_rows": int(len(df)),
            "n_groups": int(df["_group_col"].nunique()),
            "n_features": int(len(self.exog.columns)),
            "has_random_effects": bool(self.random_effects is not None),
            "est_re_cov": bool(est_re_cov),
            "ridge_alpha": float(self.ridge_alpha),
            "random_alpha": float(self.random_alpha),
            "min_group_rows_for_random_effect": int(self.min_group_rows_for_random_effect),
        }

        return MixedLinearResults(
            parameters=MixedLinearParameters(
                fe_params=fe_params,
                random_effects_by_group=random_effects_by_group,
            ),
            model_type="mixed_linear",
            train_results_df=train_results_df,
            metadata=metadata,
        )