from typing import List, Union

import lab as B
import numpy as np
import pytest
import tensorflow as tf

import spherical_harmonics.tensorflow  # noqa
from spherical_harmonics.fundamental_set import (
    FundamentalSystemCache,
    build_fundamental_system,
)
from spherical_harmonics.gegenbauer_polynomial import GegenbauerManualCoefficients
from spherical_harmonics.spherical_harmonics import (
    SphericalHarmonics,
    SphericalHarmonicsLevel,
)
from spherical_harmonics.utils import (
    spherical_to_cartesian,
    spherical_to_cartesian_4d,
    surface_area_sphere,
)


@pytest.mark.parametrize("max_degree", list(range(1, 10, 3)) + [34])
def test_orthonormal_basis_3d(max_degree):
    """Numerical check that int_{S^2} Y_i(x) Y_j(x) dx = dirac(i,j)"""

    num_grid = 300
    dimension = 3

    theta = np.linspace(0, 2 * np.pi, num_grid)  # [N]
    phi = np.linspace(0, np.pi, num_grid)  # [N]
    theta, phi = np.meshgrid(theta, phi)  # [N, N], [N, N]
    x_spherical = np.c_[theta.reshape(-1, 1), phi.reshape(-1, 1)]  # [N^2, 2]
    x_cart = spherical_to_cartesian(x_spherical)

    harmonics = SphericalHarmonics(dimension, max_degree)
    harmonics_at_x = tf.transpose(harmonics(x_cart)).numpy()  # [M, N^2]

    d_x_spherical = 2 * np.pi**2 / num_grid**2
    inner_products = (
        harmonics_at_x
        # sin(phi) to account for the surface area element of S^2
        @ (harmonics_at_x.T * np.sin(x_spherical[:, [1]]))
        * d_x_spherical
    )

    inner_products = inner_products / surface_area_sphere(dimension)

    np.testing.assert_array_almost_equal(
        inner_products, np.eye(len(harmonics_at_x)), decimal=1
    )


@pytest.mark.parametrize("max_degree", range(1, 8, 3))
def test_orthonormal_basis_4d(max_degree):
    """Numerical check that int_{S^3} Y_i(x) Y_j(x) dx = dirac(i,j)"""
    num_grid = 25
    dimension = 4

    theta1 = np.linspace(0, 2 * np.pi, num_grid)
    theta2 = np.linspace(0, np.pi, num_grid)
    theta3 = np.linspace(0, np.pi, num_grid)
    theta1, theta2, theta3 = np.meshgrid(theta1, theta2, theta3)
    x_spherical = np.c_[
        theta1.reshape((-1, 1)),
        theta2.reshape((-1, 1)),
        theta3.reshape((-1, 1)),
    ]  # [N^3, 3]
    x_cart = spherical_to_cartesian_4d(x_spherical)

    harmonics = SphericalHarmonics(dimension, max_degree)
    harmonics_at_x = tf.transpose(harmonics(x_cart))  # [M, N^3]

    d_x_spherical = 2 * np.pi**3 / num_grid**3

    inner_products = np.ones((len(harmonics_at_x), len(harmonics_at_x))) * np.nan

    for i, Y1 in enumerate(harmonics_at_x):
        for j, Y2 in enumerate(harmonics_at_x):
            v = np.sum(
                Y1
                * Y2
                # account for surface area element of S^3 sphere
                * np.sin(x_spherical[:, -1]) ** 2
                * np.sin(x_spherical[:, -2])
                * d_x_spherical
            )
            inner_products[i, j] = v

    inner_products = inner_products / surface_area_sphere(dimension)

    np.testing.assert_array_almost_equal(
        inner_products, np.eye(len(harmonics)), decimal=1
    )


@pytest.mark.parametrize("dimension", range(3, 11, 3))
@pytest.mark.parametrize("max_degree", range(2, 7, 3))
def test_equality_spherical_harmonics_collections(dimension, max_degree):

    fast_harmonics = SphericalHarmonics2(dimension, max_degree)
    harmonics = SphericalHarmonics(dimension, max_degree)

    num_points = 100
    X = np.random.randn(num_points, dimension)
    # make unit vectors
    X /= np.sum(X**2, axis=-1, keepdims=True) ** 0.5

    np.testing.assert_array_almost_equal(
        fast_harmonics(X),
        harmonics(X),
    )


@pytest.mark.parametrize("dimension", range(3, 7, 1))
@pytest.mark.parametrize("degree", range(1, 7, 3))
def test_addition_theorem(dimension, degree):
    fundamental_system = FundamentalSystemCache(dimension)
    harmonics = SphericalHarmonicsLevel(dimension, degree, fundamental_system)
    X = np.random.randn(100, dimension)
    X = X / (np.sum(X**2, keepdims=True, axis=1) ** 0.5)
    harmonics_at_X = harmonics(X)[..., None]  # [M:=N(dimension, degree), N, 1]
    harmonics_xxT = tf.matmul(
        harmonics_at_X, harmonics_at_X, transpose_b=True
    )  # [M, N, N]

    # sum over all harmonics in the level
    # addition_manual = harmonics_at_X.T @ harmonics_at_X  # [N, N]
    addition_manual = tf.reduce_sum(harmonics_xxT, axis=0)  # [N, N]
    addition_theorem = harmonics.addition(X)

    np.testing.assert_array_almost_equal(addition_manual, addition_theorem)

    np.testing.assert_array_almost_equal(
        np.diag(addition_manual)[..., None], harmonics.addition_at_1(X)
    )


@pytest.mark.parametrize("dimension", range(3, 21))
def test_init_spherical_harmonics(dimension):
    max_degree = 2
    _ = SphericalHarmonics(dimension, max_degree)


@pytest.mark.parametrize("dimension", range(21, 25))
def test_init_spherical_harmonics_not_cached(dimension):
    max_degree = 2
    with pytest.raises(ValueError):
        _ = SphericalHarmonics(dimension, max_degree)


def test_building_fundamental_set_shapes():
    dimension = 5
    x_system1 = build_fundamental_system(dimension, degree=1)
    assert x_system1.shape == (5, 5)
    x_system2 = build_fundamental_system(dimension, degree=0)
    assert x_system2.shape == (1, 5)


class SphericalHarmonics2(SphericalHarmonics):
    """
    Slightly faster implementation (approx 10%) of the `__call__` method than
    the one in `SphericalHarmonicsCollection` as we don't make use of a `map`.
    """

    def __init__(
        self, dimension: int, degrees: Union[int, List[int]], debug: bool = True
    ):
        """
        :param dimension: if d = dimension, then
            S^{d-1} = { x ∈ R^d and ||x||_2 = 1 }
            For a circle d=2, for a ball d=3
        :param degrees: list of degrees of spherical harmonic levels,
            if integer all levels (or degrees) up to `degrees` are used.
        :param debug: print debug messages.
        """
        super().__init__(dimension, degrees, debug)

        # Hack: overwrite levels to use `GegenbauerManualCoefficients`
        for harmonic in self.harmonic_levels:
            harmonic.gegenbauer = GegenbauerManualCoefficients(
                harmonic.gegenbauer.n, harmonic.gegenbauer.alpha
            )

        max_power = int(max(max(h.gegenbauer.powers) for h in self.harmonic_levels))
        self.num_harmonics = sum(len(h) for h in self.harmonic_levels)

        weights = np.zeros((self.num_harmonics, max_power + 1))  # [M, P]
        powers = np.zeros((self.num_harmonics, max_power + 1))  # [M, P]
        begin = 0
        for harmonic in self.harmonic_levels:
            coeffs = harmonic.gegenbauer.coefficients
            pows = harmonic.gegenbauer.powers
            for c, p in zip(coeffs, pows):
                for i in range(len(harmonic)):
                    weights[begin + i, int(p)] = c
                    powers[begin + i, int(p)] = p
            begin += len(harmonic)

        self.weights = tf.convert_to_tensor(weights)  # [M, P]
        self.powers = tf.convert_to_tensor(powers)  # [M, P]

        self.V = tf.convert_to_tensor(
            np.concatenate([harmonic.V for harmonic in self.harmonic_levels], axis=0)
        )  # [M, D]

        self.L_inv = tf.linalg.LinearOperatorBlockDiag(
            [
                tf.linalg.LinearOperatorFullMatrix(harmonic.L_inv)
                for harmonic in self.harmonic_levels
            ]
        )  # [M, M] block diagonal

    def __call__(self, X: B.Numeric) -> B.Numeric:
        """
        Evaluates each of the spherical harmonics in the collection,
        and stacks the results.
        :param x: TensorType, [N, D]
            N points with unit norm in cartesian coordinate system.
        :return: [num harmonics in collection, N]
        """
        VXT = tf.matmul(self.V, X, transpose_b=True)  # [M, N, 1]
        tmp = self.weights[:, None, :] * (
            VXT[:, :, None] ** self.powers[:, None, :]
        )  # [M, N, P]
        gegenbauer_at_VXT = tf.reduce_sum(tmp, axis=-1)  # [M, N]
        return tf.transpose(self.L_inv.matmul(gegenbauer_at_VXT))  # [N, M]
