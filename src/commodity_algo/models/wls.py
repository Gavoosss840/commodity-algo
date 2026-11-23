"""Weighted least squares with a forgetting factor (strategy section 1.2).

    beta = argmin sum_t w_t (y_t - a - x_t' beta)^2 + sum_j k_j (beta_j - beta0_j)^2
    w_t  = lambda^(T-t)

The second term shrinks toward the betas of the study (beta0) and is how the
algorithm "moves forward": with little data it trusts the study, and as daily
data accumulates the data progressively takes over. k_j is expressed in
pseudo-observations (config model.prior_strength_obs), so it is scale-free.

Standard errors: Newey-West (HAC), as required by the protocol (step 6).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def daily_lambda(weekly_lambda: float) -> float:
    return float(weekly_lambda) ** (1.0 / 5.0)


def forgetting_weights(n: int, lam: float) -> np.ndarray:
    return lam ** np.arange(n - 1, -1, -1, dtype=float)


@dataclass
class WLSResult:
    intercept: float
    beta: pd.Series
    se: pd.Series
    resid_sd: float
    r2: float
    n_obs: int
    fitted: pd.Series
    resid: pd.Series
    mode: str  # "prior_anchored" or "reestimated"
    alerts: list[str] = field(default_factory=list)

    def predict(self, x: pd.Series) -> float:
        b = self.beta.reindex(x.index).fillna(0.0)
        return float(self.intercept + (x * b).sum())

    def contributions(self, x: pd.Series) -> pd.Series:
        return x * self.beta.reindex(x.index).fillna(0.0)


def _newey_west_se(Xw: np.ndarray, ew: np.ndarray, XtWX_inv: np.ndarray) -> np.ndarray:
    n = len(ew)
    lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    u = Xw * ew[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov = XtWX_inv @ S @ XtWX_inv
    return np.sqrt(np.clip(np.diag(cov), 0, None))


def fit_wls(y: pd.Series, X: pd.DataFrame, lam: float = 1.0,
            prior: pd.Series | None = None, prior_strength_obs: float = 0.0,
            fix_betas: bool = False) -> WLSResult:
    """Fit y = a + X beta.

    fix_betas=True  -> betas frozen at `prior`, only the intercept is estimated
                       (weighted mean of y - X prior): the "prior_anchored" mode used
                       until enough data has accumulated.
    """
    df = pd.concat([y.rename("_y"), X], axis=1).dropna()
    yv = df["_y"].to_numpy(float)
    Xv = df[X.columns].to_numpy(float)
    n, k = Xv.shape
    w = forgetting_weights(n, lam)
    w = w / w.mean()
    cols = list(X.columns)
    prior = (prior.reindex(cols).fillna(0.0) if prior is not None else pd.Series(0.0, index=cols))

    if fix_betas or n < k + 5:
        b = prior.to_numpy(float)
        a = float(np.sum(w * (yv - Xv @ b)) / w.sum())
        resid = yv - a - Xv @ b
        se = pd.Series(np.nan, index=cols)
        mode = "prior_anchored"
    else:
        Z = np.column_stack([np.ones(n), Xv])
        W = w[:, None]
        A = Z.T @ (Z * W)
        rhs = Z.T @ (yv * w)
        if prior_strength_obs > 0:
            # Daily level residuals are strongly autocorrelated: n daily rows carry the
            # information of only n_eff = n (1-rho)/(1+rho) independent ones. Scale the
            # prior up accordingly so the data only overrides the study when it has
            # genuinely accumulated enough independent evidence.
            c0 = np.linalg.lstsq(Z * np.sqrt(W), yv * np.sqrt(w), rcond=None)[0]
            e0 = yv - Z @ c0
            rho = float(np.clip(np.corrcoef(e0[1:], e0[:-1])[0, 1], 0.0, 0.995)) if n > 10 else 0.0
            ratio = (1 + rho) / (1 - rho)
            xvar = np.average((Xv - np.average(Xv, axis=0, weights=w)) ** 2, axis=0, weights=w)
            kj = prior_strength_obs * ratio * np.where(xvar > 0, xvar, 1.0)
            A[1:, 1:] += np.diag(kj)
            rhs[1:] += kj * prior.to_numpy(float)
        A_inv = np.linalg.pinv(A)
        coef = A_inv @ rhs
        a, b = float(coef[0]), coef[1:]
        resid = yv - Z @ coef
        sw = np.sqrt(w)
        se_all = _newey_west_se(Z * sw[:, None], resid * sw, A_inv)
        se = pd.Series(se_all[1:], index=cols)
        mode = "reestimated"

    ybar = np.average(yv, weights=w)
    ss_tot = np.sum(w * (yv - ybar) ** 2)
    r2 = 1.0 - np.sum(w * resid ** 2) / ss_tot if ss_tot > 0 else np.nan
    resid_sd = float(np.sqrt(np.sum(w * resid ** 2) / max(w.sum() - (1 if mode == "prior_anchored" else k + 1), 1)))
    idx = df.index
    return WLSResult(intercept=a, beta=pd.Series(b, index=cols), se=se, resid_sd=resid_sd,
                     r2=float(r2), n_obs=n, fitted=pd.Series(yv - resid, index=idx),
                     resid=pd.Series(resid, index=idx), mode=mode)


def stability_alerts(res: WLSResult, prior: pd.Series, prior_se: pd.Series,
                     k_sigma: float = 2.0, residual_ratio: float = 1.5,
                     recent: int = 60) -> list[str]:
    """Protocol step 7: alert when a beta flips sign or leaves its historical band,
    and when recent residuals become much larger than usual (regime change)."""
    alerts = []
    if res.mode == "reestimated":
        for f, b in res.beta.items():
            b0, s0 = prior.get(f, np.nan), prior_se.get(f, np.nan)
            if np.isnan(b0):
                continue
            if np.sign(b) != np.sign(b0) and abs(b0) > 0:
                alerts.append(f"SIGN FLIP {f}: study {b0:+.3f} -> now {b:+.3f}")
            elif not np.isnan(s0) and abs(b - b0) > k_sigma * s0:
                alerts.append(f"BETA DRIFT {f}: study {b0:+.3f} +/- {s0:.3f} -> now {b:+.3f}")
    if len(res.resid) > recent * 3:
        r_recent = res.resid.iloc[-recent:].std()
        r_long = res.resid.std()
        if r_long > 0 and r_recent > residual_ratio * r_long:
            alerts.append(f"RESIDUALS WIDENING: last {recent}d sd {r_recent:.4f} vs {r_long:.4f} -> signal suspended")
    return alerts
