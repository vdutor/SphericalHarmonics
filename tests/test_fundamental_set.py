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

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import linalg
from scipy.special import gegenbauer as ScipyGegenbauer

from spherical_harmonics.fundamental_set import (
    FundamentalSystemCache,
    calculate_decrement_in_determinant,
    grad_calculate_decrement_in_determinant,
)
from spherical_harmonics.gegenbauer_polynomial import Gegenbauer


def det(Z, X_system, C_old, gegenbauer: Gegenbauer):
    """
    Objective to

    Z: [d]
    X_system: [N, d]
    C_old: [N, N]
    """
    norm = tf.reduce_sum(Z**2) ** 0.5  # scalar
    X = Z / norm  # [d]
    XXd = tf.einsum("nd,d->n", X_system, X)  # [N]
    M_cross = gegenbauer(XXd)  # [N]

    # Determinant is given by the compuations below though we only compute the
    # bits that depend on Z and make sure the objective can be minimised in
    # order to maximise the overall determinant:
    # C_1 = gegenbauer.value_at_1
    # det_C_old = tf.linalg.det(C_old)
    # return det_C_old * (C_1 - tf.reduce_sum(res ** 2))

    C_old_chol = tf.linalg.cholesky(C_old)
    res = tf.linalg.triangular_solve(C_old_chol, M_cross[:, None], lower=True)
    return tf.reduce_sum(res**2)


@pytest.mark.parametrize("dimension", [3, 5, 6, 9])
@pytest.mark.parametrize("degree", [5, 4, 5])
def test_objective(dimension, degree):
    alpha = (dimension - 2) / 2
    gegenbauer = ScipyGegenbauer(degree, alpha)
    system = FundamentalSystemCache(dimension=dimension)
    X_system = system.load(degree)
    C_new = gegenbauer(X_system @ X_system.T)
    Z = X_system[-1]
    X_system = X_system[:-1]
    C_old = gegenbauer(X_system @ X_system.T)
    det1 = linalg.det(C_new)

    C_1 = gegenbauer(1.0)
    det_C_old = tf.linalg.det(C_old)
    v = calculate_decrement_in_determinant(
        Z, X_system, linalg.cholesky(C_old, lower=True), gegenbauer
    )
    det2 = det_C_old * (C_1 - v)
    np.testing.assert_allclose(det1, det2)


@pytest.mark.parametrize("dimension", [3, 5, 6, 9])
@pytest.mark.parametrize("degree", [3, 4, 5])
def test_grad_objective(dimension, degree):
    alpha = (dimension - 2) / 2
    gegenbauer = Gegenbauer(degree, alpha)
    system = FundamentalSystemCache(dimension=dimension)
    X_system = system.load(degree)
    X_system = tf.convert_to_tensor(X_system, dtype=tf.float64)
    Z = tf.random.normal((dimension,), dtype=tf.float64)
    X_system = X_system[:-1]
    C_old = gegenbauer(tf.matmul(X_system, X_system, transpose_b=True))

    _, dv1 = tfp.math.value_and_gradient(
        lambda Z: det(Z, X_system, C_old, gegenbauer), Z
    )
    dv1 = dv1.numpy()
    dv2 = grad_calculate_decrement_in_determinant(
        Z.numpy(),
        X_system.numpy(),
        linalg.cholesky(C_old.numpy(), lower=True),
        ScipyGegenbauer(degree, alpha),
    )
    np.testing.assert_array_almost_equal(dv1, dv2)
