"""Base-parameter extraction from rank-deficient regressors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _pivot_columns_fallback(
    matrix: NDArray[np.float64],
    tolerance: float,
) -> tuple[int, NDArray[np.int64]]:
    """Greedy column-pivot selection using orthogonal residual norms."""

    n_columns = matrix.shape[1]
    remaining = list(range(n_columns))
    selected: list[int] = []
    orthonormal_basis = np.zeros((matrix.shape[0], 0), dtype=float)

    while remaining:
        best_index = remaining[0]
        best_norm = -np.inf
        best_residual = None
        for column in remaining:
            residual = matrix[:, column].astype(float, copy=True)
            if orthonormal_basis.shape[1]:
                residual -= orthonormal_basis @ (orthonormal_basis.T @ residual)
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm > best_norm:
                best_index = column
                best_norm = residual_norm
                best_residual = residual
        if best_residual is None or best_norm <= tolerance:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        orthonormal_basis = np.column_stack(
            [orthonormal_basis, best_residual / best_norm]
        )

    pivots = np.asarray(selected + remaining, dtype=int)
    return len(selected), pivots


@dataclass(frozen=True)
class BaseParameterMapping:
    rank: int
    permutation: NDArray[np.float64]
    beta: NDArray[np.float64]
    pivot_indices: NDArray[np.int64]

    @property
    def num_base_params(self) -> int:
        return int(self.rank)

    @property
    def num_dep_params(self) -> int:
        return int(self.permutation.shape[1] - self.rank)

    @property
    def permutation_matrix(self) -> NDArray[np.float64]:
        return self.permutation

    def base_projection(self) -> NDArray[np.float64]:
        return self.permutation[:, : self.rank]

    def dependent_projection(self) -> NDArray[np.float64]:
        return self.permutation[:, self.rank :]

    def sort_parameters(self, parameters: ArrayLike) -> NDArray[np.float64]:
        vector = np.asarray(parameters, dtype=float).reshape(-1)
        if vector.shape != (self.permutation.shape[0],):
            raise ValueError("parameter vector has incompatible size")
        return self.permutation.T @ vector

    def base_parameter_vector(self, parameters: ArrayLike) -> NDArray[np.float64]:
        sorted_parameters = self.sort_parameters(parameters)
        independent = sorted_parameters[: self.rank]
        dependent = sorted_parameters[self.rank :]
        if dependent.size == 0:
            return independent.copy()
        return independent + self.beta @ dependent

    def reconstruction_matrix(self) -> NDArray[np.float64]:
        total_parameters = self.permutation.shape[0]
        inverse = np.eye(total_parameters, dtype=float)
        if self.num_dep_params:
            inverse[: self.rank, self.rank :] = -self.beta
        return self.permutation @ inverse

    def reconstruct_full_parameters(
        self,
        base_parameters: ArrayLike,
        dependent_parameters: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        base = np.asarray(base_parameters, dtype=float).reshape(-1)
        if base.shape != (self.rank,):
            raise ValueError("base_parameters has incompatible size")
        if dependent_parameters is None:
            dependent = np.zeros(self.num_dep_params, dtype=float)
        else:
            dependent = np.asarray(dependent_parameters, dtype=float).reshape(-1)
            if dependent.shape != (self.num_dep_params,):
                raise ValueError("dependent_parameters has incompatible size")
        return self.reconstruction_matrix() @ np.concatenate([base, dependent])


def extract_base_parameters(
    observation_matrix: ArrayLike,
    *,
    tolerance: float | None = None,
) -> BaseParameterMapping:
    """Find independent and dependent columns using pivoted QR."""

    try:
        from scipy.linalg import qr
    except Exception:
        qr = None

    matrix = np.asarray(observation_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("observation_matrix must be 2D")
    if qr is None:
        if tolerance is None:
            tolerance = np.finfo(float).eps * max(matrix.shape) * np.linalg.norm(
                matrix,
                ord=2,
            )
        rank, pivots = _pivot_columns_fallback(matrix, tolerance)
        permutation = np.eye(matrix.shape[1])[:, pivots]
        if rank == 0 or rank == matrix.shape[1]:
            beta = np.zeros((rank, matrix.shape[1] - rank), dtype=float)
        else:
            base_columns = matrix @ permutation[:, :rank]
            dependent_columns = matrix @ permutation[:, rank:]
            beta, *_ = np.linalg.lstsq(base_columns, dependent_columns, rcond=None)
            beta[np.abs(beta) < np.sqrt(np.finfo(float).eps)] = 0.0
        return BaseParameterMapping(
            rank=rank,
            permutation=permutation,
            beta=beta,
            pivot_indices=pivots,
        )
    else:
        _, r, pivots = qr(matrix, pivoting=True, mode="economic")
    diag = np.abs(np.diag(r))
    if tolerance is None:
        tolerance = np.finfo(float).eps * max(matrix.shape) * (diag[0] if diag.size else 0.0)
    rank = int(np.sum(diag > tolerance))
    permutation = np.eye(matrix.shape[1])[:, pivots]

    if rank == 0 or rank == matrix.shape[1]:
        beta = np.zeros((rank, matrix.shape[1] - rank), dtype=float)
    else:
        r1 = r[:rank, :rank]
        r2 = r[:rank, rank:]
        beta = np.linalg.solve(r1, r2)
        beta[np.abs(beta) < np.sqrt(np.finfo(float).eps)] = 0.0

    return BaseParameterMapping(
        rank=rank,
        permutation=permutation,
        beta=beta,
        pivot_indices=np.asarray(pivots, dtype=int),
    )
