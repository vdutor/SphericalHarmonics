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

import math

import lab as B
import numpy as np
from beartype.typing import List, Tuple, Union
from lab import dispatch
from scipy.special import gegenbauer as scipy_gegenbauer
from scipy.special import loggamma

from spherical_harmonics.lab_extras import polyval


class Polynomial:
    r"""
    One-dimensional polynomial function expressed with coefficients and powers.
    The polynomial f(x) is given by f(x) = \sum_i c_i x^{p_i}, with x \in R^1.
    """

    def __init__(
        self,
        coefficients: Union[List, np.ndarray],
        powers: Union[List, np.ndarray],
    ):
        r"""
        The polynomial f(x) is given by f(x) = \sum_i c_i x^{p_i},
        with c coefficients and p powers. The number of coefficients and
        the number of powers must match.

        :param coefficients: list of weights
        :param powers: list of powers
        """
        assert len(coefficients) == len(powers)
        self.coefficients = coefficients
        self.powers = powers

    def __call__(self, x: B.Numeric) -> B.Numeric:
        """
        Evaluates the polynomial @ `x`
        :param x: 1D input values at which to evaluate the polynomial, [...]

        :return:
            function evaluations, same shape as `x` [...]
        """
        cs = B.reshape(self.coefficients, 1, -1)  # [1, M]
        ps = B.reshape(self.powers, 1, -1)  # [1, M]
        x_flat = B.reshape(x, -1, 1)  # [N, 1]
        val = B.sum(cs * (x_flat**ps), axis=1)  # [N, M]  # [N]
        return B.reshape(val, *B.shape(x))


class GegenbauerManualCoefficients(Polynomial):
    r"""
    Gegenbauer polynomials or ultraspherical polynomials C_n^(α)(x)
    are orthogonal polynomials on the interval [−1,1] with respect
    to the weight function (1 − x^2)^{α–1/2} [1].

    [1] https://en.wikipedia.org/wiki/Gegenbauer_polynomials,
    [2] Approximation Theory and Harmonic Analysis on Spheres and Balls,
        Feng Dai and Yuan Xu, Chapter 1. Spherical Harmonics,
        https://arxiv.org/pdf/1304.2585.pdf
    """

    def __init__(self, n: int, alpha: float):
        """
        Gegenbauer polynomial C_n^(α)(z) of degree `n` and characterisation `alpha`.
        We represent the Gegenbauer function as a polynomial and compute its
        coefficients and corresponding powers.

        :param n: degree
        :param alpha: characterises the form of the polynomial.
            Typically changes with the dimension, alpha = (dimension - 2) / 2
        """

        coefficients, powers = self._compute_coefficients_and_powers(n, alpha)
        super().__init__(
            np.array(coefficients, dtype=np.float64),
            np.array(powers, dtype=np.float64),
        )
        self.n = n
        self.alpha = alpha
        self._at_1 = scipy_gegenbauer(self.n, self.alpha)(1.0)

    def _compute_coefficients_and_powers(
        self, n: int, alpha: float
    ) -> Tuple[List, List]:
        """
        Compute the weights (coefficients) and powers of the Gegenbauer functions
        expressed as polynomial.

        :param n: degree
        :param alpha:
        """
        coefficients, powers = [], []

        for k in range(math.floor(n / 2) + 1):  # k=0 .. floor(n/2) (incl.)
            # computes the coefficients in log space for numerical stability
            log_coef = loggamma(n - k + alpha)
            log_coef -= loggamma(alpha) + loggamma(k + 1) + loggamma(n - 2 * k + 1)
            log_coef += (n - 2 * k) * np.log(2)
            coef = np.exp(log_coef)
            coef *= (-1) ** k
            coefficients.append(coef)
            powers.append(n - 2 * k)

        return coefficients, powers

    def __call__(self, x: B.Numeric) -> B.Numeric:
        if self.n < 0:
            return B.zeros(x)
        elif self.n == 0:
            return B.ones(x)
        elif self.n == 1:
            return 2 * self.alpha * x
        else:
            return super().__call__(x)

    @property
    def value_at_1(self):
        """
        Gegenbauer evaluated at 1.0
        """
        return self._at_1


class GegenbauerScipyCoefficients:
    """Gegenbauer polynomial using the coefficients given by Scipy."""

    def __init__(self, n: int, alpha: float):
        self.n = n
        self.alpha = alpha
        self.C = scipy_gegenbauer(n, alpha)
        self._at_1 = self.C(1.0)
        self.coefficients = list(self.C.coefficients)

    @dispatch
    def __call__(self, x: B.NPNumeric) -> B.Numeric:
        if self.n < 0:
            return B.zeros(x)
        elif self.n == 0:
            return B.ones(x)
        elif self.n == 1:
            return 2 * self.alpha * x

        return self.C(x)

    @dispatch  # type: ignore[no-redef]
    def __call__(self, x: B.Numeric) -> B.Numeric:
        """x: [...], return [...]"""
        if self.n < 0:
            return B.zeros(x)
        elif self.n == 0:
            return B.ones(x)
        elif self.n == 1:
            return 2 * self.alpha * x

        return polyval(self.coefficients, x)

    @property
    def value_at_1(self):
        """Gegenbauer evaluated at 1.0"""
        return self._at_1


Gegenbauer = GegenbauerScipyCoefficients
