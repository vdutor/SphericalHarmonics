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

""" Spherical Harmonics and associated utility functions """

import lab as B
import numpy as np
from beartype.typing import List, Union
from scipy.special import gegenbauer as scipy_gegenbauer

from spherical_harmonics.fundamental_set import FundamentalSystemCache, num_harmonics
from spherical_harmonics.gegenbauer_polynomial import Gegenbauer
from spherical_harmonics.lab_extras import from_numpy
from spherical_harmonics.utils import surface_area_sphere


class SphericalHarmonics:
    r"""
    Contains all the spherical harmonic levels up to a `max_degree`.
    Total number of harmonics in the collection is given by
    :math:`\sum_{degree=0}^{max_degree-1} num_harmonics(dimension, degree)`
    """

    def __init__(
        self,
        dimension: int,
        degrees: Union[int, List[int]],
        debug: bool = False,
        allow_uncomputed_levels: bool = False,
    ):
        """
        :param dimension: if d = dimension, then
            S^{d-1} = { x ∈ R^d and ||x||_2 = 1 }
            For a circle d=2, for a ball d=3
        :param degrees: list of degrees of spherical harmonic levels,
            if integer all levels (or degrees) up to `degrees` are used.
        highest degree of polynomial
            in the collection (exclusive)
        :param allow_uncomputed_levels: if True, allow levels without the precomputed
            fundamental system.
        :param debug: print debug messages.
        """
        assert (
            dimension >= 3
        ), f"Lowest supported dimension is 3, you specified {dimension}"
        self.debug = debug

        if isinstance(degrees, int):
            degrees = list(range(degrees))

        self.fundamental_system = FundamentalSystemCache(
            dimension, strict_loading=not allow_uncomputed_levels
        )
        self.harmonic_levels = [
            SphericalHarmonicsLevel(dimension, degree, self.fundamental_system)
            for degree in degrees
        ]

    def __call__(
        self,
        x: B.Numeric,
    ) -> B.Numeric:
        """
        Evaluates each of the spherical harmonic level in the collection,
        and stacks the results.
        :param x: [N, D]
            N points with unit norm in cartesian coordinate system.
        :return: [N, num harmonics in collection]
        """
        values = map(
            lambda harmonic: harmonic(x), self.harmonic_levels
        )  # List of length `max_degree` with Tensor [num_harmonics_degree, N]

        return B.transpose(B.concat(*list(values), axis=0))  # [num_harmonics, N]

    def __len__(self):
        return sum(len(harmonic_level) for harmonic_level in self.harmonic_levels)

    def num_levels(self):
        return len(self.harmonic_levels)

    def addition(self, X, X2=None):
        """For test purposes only"""
        return B.sum(
            B.stack(
                *[level.addition(X, X2) for level in self.harmonic_levels],
                axis=0,
            ),
            axis=0,
        )  # [N1, N2]


class SphericalHarmonicsLevel:
    r"""
    Spherical harmonics \phi(x) in a specific dimension and degree (or level).

    The harmonics are constructed by
    1) Building a fundamental set of directions {v_i}_{i=1}^N,
        where N is number of harmonics of the degree.
        Given these directions we have that {c(<v_i, x>)}_i is a basis,
        where c = gegenbauer(degree, alpha) and alpha = (dimension - 2)/2.
        See Definition 3.1 in [1].
    2) Using Gauss Elimination we orthogonalise this basis, which
       corresponds to the Gram-Schmidt procedure.

    [1] Approximation Theory and Harmonic Analysis on Spheres and Balls,
        Feng Dai and Yuan Xu, Chapter 1. Spherical Harmonics,
        https://arxiv.org/pdf/1304.2585.pdf
    """

    def __init__(self, dimension: int, degree: int, fundamental_system=None):
        r"""
        param dimension: if d = dimension, then
            S^{d-1} = { x ∈ R^d and ||x||_2 = 1 }
            For a circle d=2, for a ball d=3
        param degree: degree of the harmonic, also referred to as level.
        """
        assert (
            dimension >= 3
        ), f"Lowest supported dimension is 3, you specified {dimension}"
        self.dimension, self.degree = dimension, degree
        self.alpha = (self.dimension - 2) / 2.0
        self.num_harmonics_in_level = num_harmonics(self.dimension, self.degree)

        self.V = fundamental_system.load(self.degree)

        # surface area of S^{d−1}
        self.surface_area_sphere = surface_area_sphere(dimension)
        # normalising constant
        c = self.alpha / (degree + self.alpha)

        if self.is_level_computed:
            VtV = np.dot(self.V, self.V.T)
            self.A = c * scipy_gegenbauer(self.degree, self.alpha)(VtV)
            self.L = np.linalg.cholesky(self.A)  # [M, M]
            # Cholesky inverse corresponds to the weights you get from Gram-Schmidt
            self.L_inv = np.linalg.solve(self.L, np.eye(len(self.L)))
        self.gegenbauer = Gegenbauer(self.degree, self.alpha)

    @property
    def is_level_computed(self) -> bool:
        """
        Whether the level has the fundamental system computed.
        """
        return self.V is not None

    def __call__(self, X: B.Numeric) -> B.Numeric:
        r"""
        :param X: M normalised (i.e. unit) D-dimensional vector, [N, D]

        :return: `X` evaluated at the M spherical harmonics in the set.
            [\phi_m(x_i)], shape [M, N]
        """
        if not self.is_level_computed:
            raise ValueError(
                f"Fundamental system for dimension {self.dimension} and degree "
                f"{self.degree} has not been precomputed. Terminating "
                "computations. Precompute set by running `fundamental_set.py`"
            )

        VXT = B.matmul(
            B.cast(B.dtype(X), from_numpy(X, self.V)), X, tr_b=True
        )  # [M, N]
        zonals = self.gegenbauer(VXT)  # [M, N]
        return B.matmul(B.cast(B.dtype(X), self.L_inv), zonals)  # [M, N]

    # TODO(Vincent) for some reason Optional[B.Numeric] doesn't work
    def addition(self, X: B.Numeric, Y: B.Numeric = None) -> B.Numeric:
        r"""
        Addition theorem. The sum of the product of all the spherical harmonics evaluated
        at x and x' of a specific degree simplifies to the gegenbauer polynomial evaluated
        at the inner product between x and x'.

        Mathematically:
            \sum_{k=1}^{N(dim, degree)} \phi_k(X) * \phi_k(Y)
                = (degree + \alpha) / \alpha * C_degree^\alpha(X^T Y)
        where \alpha = (dimension - 2) / 2 and omega_d the surface area of the
        hypersphere S^{d-1}.

        :param X: Unit vectors on the (hyper) sphere [N1, D]
        :param Y: Unit vectors on the (hyper) sphere [N2, D].
            If None, X is used as Y.

        :return: [N1, N2]
        """
        if Y is None:
            Y = X
        XYT = B.matmul(X, Y, tr_b=True)  # [N1, N2]
        c = self.gegenbauer(XYT)  # [N1, N2]
        return (self.degree / self.alpha + 1.0) * c  # [N1, N2]

    def addition_at_1(self, X: B.Numeric) -> B.Numeric:
        r"""
        Evaluates \sum_k \phi_k(x) \phi_k(x), notice the argument at which we evaluate
        the harmonics is equal. See `self.addition` for the general case.

        This simplifies to
            \sum_{k=1}^{N(dim, degree)} \phi_k(x) * \phi_k(x)
                = (degree + \alpha) / \alpha * C_degree^\alpha(1)

        as all vectors in `X` are normalised so that x^\top x == 1.

        :param X: only used for it's X.shape[0], [N, D]
        :return: [N, 1]
        """
        c = (
            B.ones(
                X.dtype,
                *(X.shape[0], 1),
            )
            * self.gegenbauer.value_at_1
        )  # [N, 1]
        return (self.degree / self.alpha + 1.0) * c  # [N, 1]

    def eigenvalue(self) -> Union[int, float, np.ndarray]:
        """
        Spherical harmonics are eigenfunctions of the Laplace-Beltrami operator
        (also known as the Spherical Laplacian). We return the associated
        eigenvalue.

        The eigenvalue of the N(dimension, degree) number of spherical harmonics
        on the same level (i.e. same degree) is the same.
        """
        return eigenvalue_harmonics(self.degree, self.dimension)

    def __len__(self):
        return self.num_harmonics_in_level


def eigenvalue_harmonics(
    degrees: Union[int, float, np.ndarray], dimension: int
) -> Union[int, float, np.ndarray]:
    """
    Eigenvalue of a spherical harmonic of a specific degree.

    :param degrees: a single, or a array of degrees
    :param dimension:
        S^{d-1} = { x ∈ R^d and ||x||_2 = 1 }
        For a circle d=2, for a ball d=3

    :return: the corresponding eigenvalue of the spherical harmonic
        for the specified degrees, same shape as degrees.
    """
    assert dimension >= 3, "We only support dimensions >= 3"

    return degrees * (degrees + dimension - 2)
