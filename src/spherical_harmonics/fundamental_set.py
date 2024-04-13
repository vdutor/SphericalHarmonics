# Copyright 2021 Vincent Dutordoir. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from pkg_resources import resource_filename
from scipy import linalg, optimize
from scipy.special import comb as combinations
from scipy.special import gegenbauer as ScipyGegenbauer


def num_harmonics(dimension: int, degree: int) -> int:
    r"""
    Number of spherical harmonics of a particular degree n in
    d dimensions. Referred to as N(d, n).

    param dimension:
        S^{d-1} = { x ∈ R^d and ||x||_2 = 1 }
        For a circle d=2, for a ball d=3
    param degree: degree of the harmonic
    """
    if degree == 0:
        return 1
    elif dimension == 3:
        return int(2 * degree + 1)
    else:
        comb = combinations(degree + dimension - 3, degree - 1)
        return int(np.round((2 * degree + dimension - 2) * comb / degree))


class FundamentalSystemCache:
    """A simple cache object to access precomputed fundamental system.

    Fundamental system are sets of points that allow the user to evaluate the spherical
    harmonics in an arbitrary dimension"""

    def __init__(
        self,
        dimension: int,
        load_dir="fundamental_system",
        only_use_cache: bool = True,
        strict_loading: bool = True,
    ):
        self.file_name = Path(
            resource_filename(__name__, f"{load_dir}/fs_{dimension}D.npz")
        )
        self.dimension = dimension
        self.only_use_cache = only_use_cache
        self.strict_loading = strict_loading

        if self.file_name.exists():
            with np.load(self.file_name) as data:
                self.cache = {k: v for (k, v) in data.items()}
        elif only_use_cache and self.strict_loading:
            raise ValueError(
                f"Fundamental system for dimension {dimension} has not been precomputed."
                "Terminating computations. Precompute set by running `fundamental_set.py`"
            )
        else:
            self.cache = {}

    @property
    def max_computed_degree(self) -> int:
        max_degree = -1
        while True:
            max_degree = max_degree + 1
            key = self.cache_key(max_degree)
            if key not in self.cache:
                break
        return max_degree

    def cache_key(self, degree: int) -> str:
        """Return the key used in the cache"""
        return f"degree_{degree}"

    def load(self, degree: int) -> Optional[np.ndarray]:
        """Load or calculate the set for given degree"""
        key = self.cache_key(degree)
        if key not in self.cache:
            if self.only_use_cache and self.strict_loading:
                raise ValueError(
                    f"Fundamental system for dimension {self.dimension} and degree "
                    f"{degree} has not been precomputed. Terminating "
                    "computations. Precompute set by running `fundamental_set.py`"
                )
            elif not self.only_use_cache:
                print("WARNING: Cache miss - calculating  system")
                self.cache[key] = self.calculate(self.dimension, degree)

        if key not in self.cache and not self.strict_loading:
            warnings.warn(
                f"Fundamental system for dimension {self.dimension} has been "
                f"computed up to degree {self.max_computed_degree}. This means you will not "
                "be able to evaluate individual spherical harmonics for larger degrees.",
                RuntimeWarning,
            )
            return None

        return self.cache[key]

    def regenerate_and_save_cache(self, max_degrees: int) -> None:
        """Regenerate and overwrite saved cache to disk"""
        system = {}
        for degree in range(max_degrees):
            print(f"finding level {degree}/{max_degrees} in {self.dimension}D")
            d_system = self.calculate(
                self.dimension, degree, gtol=1e-8, num_restarts=10
            )
            system[f"degree_{degree}"] = d_system
        with open(self.file_name, "wb+") as f:
            np.savez(f, **system)

    @staticmethod
    def calculate(
        dimension: int, degree: int, *, gtol: float = 1e-5, num_restarts: int = 1
    ) -> np.ndarray:
        return build_fundamental_system(
            dimension, degree, gtol=gtol, num_restarts=num_restarts
        )


def build_fundamental_system(
    dimension, degree, *, gtol=1e-5, num_restarts=1, debug=False
):
    """
    We build a fundamental system incrementally, by adding a new candidate vector each
    time and maximising the span of the space generated by these spherical harmonics.

    This can be done by greedily optimising the determinant of the gegenbauered Gram matrix.

    Based on [1, Defintion 3.1]

    [1] Approximation Theory and Harmonic Analysis on Spheres and Balls,
        Feng Dai and Yuan Xu, Chapter 1. Spherical Harmonics,
        https://arxiv.org/pdf/1304.2585.pdf
    """
    alpha = (dimension - 2) / 2.0
    gegenbauer = ScipyGegenbauer(degree, alpha)
    system_size = num_harmonics(dimension, degree)

    # 1. Choose first direction in system to be north pole
    Z0 = np.eye(dimension)[-1]
    X_system = normalize(Z0).reshape(1, dimension)
    M_system_chol = cholesky_of_gegenbauered_gram(gegenbauer, X_system)

    # 2. Find a new vector incrementally by max'ing the determinant of the gegenbauered Gram
    for i in range(1, system_size):

        Z_next, ndet, restarts = None, np.inf, 0
        while restarts <= num_restarts:
            x_init = np.random.randn(dimension)
            result = optimize.fmin_bfgs(
                f=calculate_decrement_in_determinant,
                fprime=grad_calculate_decrement_in_determinant,
                x0=x_init,
                args=(X_system, M_system_chol, gegenbauer),
                full_output=True,
                gtol=gtol,
                disp=debug,
            )

            if result[1] <= ndet:
                Z_next, ndet, *_ = result
                #  TODO: we should we break when we find the best vector.
                #  Unclear how to do this at this point.
            # Try again with new x_init
            restarts += 1
        print(
            f"det: {-ndet:11.4f}, ({i + 1:5d} of {system_size:5d}, "
            f"degree {degree}: {dimension}D)"
        )
        X_next = normalize(Z_next).reshape(1, dimension)
        X_system = np.vstack([X_system, X_next])
        M_system_chol = cholesky_of_gegenbauered_gram(gegenbauer, X_system)

    return X_system


def calculate_decrement_in_determinant(Z, X_system, M_system_chol, gegenbauer):
    r"""Calculate the negative determinant.

    :param Z: is a potential vector for the next fundamental point (it will get normalized)
    :param X_system: is a matrix of existing fundamental points [num_done, D]
    :param M_system_chol: is the cholesky of the matrix M of the done points [num_done, num_done]

    :return: the negative-increment of the determinant of the matrix with Z (normalized)
     added to the done points
    """
    X = normalize(Z)
    XXd = np.dot(X_system, X)  # [num_done,]

    # M_new = gegenbauer(1.0)  # X normalized so X @ X^T = 1
    M_cross = gegenbauer(XXd)

    # Determinant of M is computed efficiently making use of the Schur complement
    # M = [[ M_system_chol, M_cross], [ M_cross^T, M_new]]
    # det(M) = det(M_system_chol) * det(M_new - M_cross^T M_system_chol^{-1} M_cross )
    res = linalg.solve_triangular(M_system_chol, M_cross, trans=0, lower=True)
    return np.sum(np.square(res))


def grad_calculate_decrement_in_determinant(Z, X_system, M_system_chol, gegenbauer):
    r"""Calculate the negative determinant.

    :param Z: is a potential vector for the next fundamental point (it will get normalized)
    :param X_system: is a matrix of existing fundamental points [num_done, D]
    :param M_system_chol: is the cholesky of the matrix M of the done points [num_done, num_done]

    """
    X = normalize(Z)
    XXd = np.dot(X_system, X)  # [num_done,]

    M_cross = gegenbauer(XXd)

    res = linalg.solve_triangular(M_system_chol, M_cross, trans=0, lower=True)
    dM_cross = 2.0 * linalg.solve_triangular(
        M_system_chol,
        res,
        trans=1,
        lower=True,
    )
    dXXd = gegenbauer.deriv()(XXd) * dM_cross
    dX = np.dot(X_system.T, dXXd)
    dZ = (dX - X * np.dot(X, dX)) / norm(Z)
    return dZ


def cholesky_of_gegenbauered_gram(gegenbauer_polynomial, x_matrix):
    XtX = x_matrix @ x_matrix.T
    return np.linalg.cholesky(gegenbauer_polynomial(XtX))


def normalize(vec: np.ndarray):
    assert len(vec.shape) == 1
    return vec / norm(vec)


def norm(vec: np.ndarray):
    assert len(vec.shape) == 1
    return np.sqrt(np.sum(np.square(vec)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-calculate fundamental system")
    parser.add_argument("-d", "--dim", default=3, type=int, help="Dimension")
    degrees_levels_group = parser.add_mutually_exclusive_group()
    degrees_levels_group.add_argument("-m", "--max-harmonics", default=1000, type=int)
    degrees_levels_group.add_argument("-l", "--max-degrees", type=int)

    args = parser.parse_args()

    def calc_degrees(dimension: int, max_harmonics: int):
        harmonics = 0
        degree = 1
        while harmonics < max_harmonics:
            harmonics += num_harmonics(dimension, degree)
            degree += 1
        degree -= 1
        return degree

    def regenerate_cache(dimension: int, max_degrees: int):
        FundamentalSystemCache(
            dimension, only_use_cache=False
        ).regenerate_and_save_cache(max_degrees)

    max_degrees = args.max_degrees or calc_degrees(args.dim, args.max_harmonics)

    regenerate_cache(args.dim, max_degrees)
