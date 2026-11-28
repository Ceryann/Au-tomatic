"""Minimal Gaussian Hidden Markov Model (diagonal covariance) fitted by EM.

Written with numpy only so the project does not depend on `hmmlearn` (which can be
hard to install on Windows). Intended for small problems: ~500-1000 daily
observations, 3 states, 2-4 features.

Notes
-----
* Forward-backward uses scaling (not log-sums) for speed; emissions are
  max-normalised per time step for numerical stability.
* `gamma_[-1]` (smoothed posterior of the LAST observation) equals the filtered
  probability, because no future observations exist. That is what we use for "today".
* Several initialisations are tried; the best log-likelihood wins. Initialisation
  splits observations by tertiles of the feature at `vol_col` (realised volatility),
  which makes the states meaningful and the result reproducible.
"""

from __future__ import annotations

import numpy as np


class GaussianHMM:
    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 150,
        tol: float = 1e-4,
        var_floor: float = 1e-3,
        n_inits: int = 4,
        seed: int = 0,
        vol_col: int = 1,
    ):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.var_floor = var_floor
        self.n_inits = max(1, n_inits)
        self.seed = seed
        self.vol_col = vol_col

        self.startprob_: np.ndarray
        self.transmat_: np.ndarray
        self.means_: np.ndarray
        self.vars_: np.ndarray
        self.loglik_: float = -np.inf
        self.gamma_: np.ndarray
        self.n_iter_run_: int = 0

    # ------------------------------------------------------------ emissions
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        T, _ = X.shape
        out = np.empty((T, self.K))
        for k in range(self.K):
            var = self.vars_[k]
            out[:, k] = -0.5 * (
                np.sum(np.log(2 * np.pi * var)) + np.sum((X - self.means_[k]) ** 2 / var, axis=1)
            )
        return out

    # ------------------------------------------------------ forward-backward
    def _forward_backward(self, logB: np.ndarray):
        T, K = logB.shape
        A = self.transmat_
        m = logB.max(axis=1, keepdims=True)
        B = np.exp(logB - m)

        alpha = np.empty((T, K))
        c = np.empty(T)
        a = self.startprob_ * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * B[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]

        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = (
            alpha[:-1, :, None] * A[None, :, :] * (B[1:] * beta[1:])[:, None, :] / c[1:, None, None]
        ).sum(axis=0)
        loglik = float(np.sum(np.log(c)) + m.sum())
        return loglik, gamma, xi

    # ---------------------------------------------------------------- M-step
    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray | None):
        nk = gamma.sum(axis=0) + 1e-10
        self.means_ = (gamma.T @ X) / nk[:, None]
        diff2 = (X[:, None, :] - self.means_[None, :, :]) ** 2          # T x K x D
        self.vars_ = np.einsum("tk,tkd->kd", gamma, diff2) / nk[:, None] + self.var_floor
        if xi is not None:
            trans = xi + 1e-3                                           # smoothing, avoids zeros
            self.transmat_ = trans / trans.sum(axis=1, keepdims=True)
            self.startprob_ = (gamma[0] + 1e-3) / (gamma[0] + 1e-3).sum()

    def _init(self, X: np.ndarray, rng: np.random.Generator, noise: float) -> None:
        T = len(X)
        key = X[:, self.vol_col] if X.shape[1] > self.vol_col else X[:, 0] ** 2
        order = np.argsort(key, kind="stable")
        resp = np.zeros((T, self.K))
        for k, idx in enumerate(np.array_split(order, self.K)):
            resp[idx, k] = 1.0
        if noise > 0:
            resp = (1 - noise) * resp + noise * rng.dirichlet(np.ones(self.K), size=T)
        self.transmat_ = np.full((self.K, self.K), 0.1 / (self.K - 1))
        np.fill_diagonal(self.transmat_, 0.9)
        self.startprob_ = np.full(self.K, 1.0 / self.K)
        self._m_step(X, resp, None)

    # ------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray) -> "GaussianHMM":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < 30:
            raise ValueError("Need a 2-D array with at least 30 observations.")
        rng = np.random.default_rng(self.seed)
        best = None
        for i in range(self.n_inits):
            self._init(X, rng, noise=0.0 if i == 0 else 0.35)
            prev = -np.inf
            ll, gamma, it = -np.inf, None, 0
            for it in range(1, self.n_iter + 1):
                ll, gamma, xi = self._forward_backward(self._log_emission(X))
                self._m_step(X, gamma, xi)
                if abs(ll - prev) < self.tol * max(1.0, abs(ll)):
                    break
                prev = ll
            # final E-step with the last parameters so gamma matches them
            ll, gamma, _ = self._forward_backward(self._log_emission(X))
            if best is None or ll > best["ll"]:
                best = {
                    "ll": ll, "gamma": gamma, "it": it,
                    "params": (self.startprob_.copy(), self.transmat_.copy(),
                               self.means_.copy(), self.vars_.copy()),
                }
        self.startprob_, self.transmat_, self.means_, self.vars_ = best["params"]
        self.loglik_, self.gamma_, self.n_iter_run_ = best["ll"], best["gamma"], best["it"]
        return self

    # ----------------------------------------------------------------- utils
    def n_params(self, n_features: int) -> int:
        k = self.K
        return (k - 1) + k * (k - 1) + 2 * k * n_features

    def bic(self, n_obs: int, n_features: int) -> float:
        return -2.0 * self.loglik_ + self.n_params(n_features) * np.log(n_obs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, gamma, _ = self._forward_backward(self._log_emission(np.asarray(X, dtype=float)))
        return gamma
