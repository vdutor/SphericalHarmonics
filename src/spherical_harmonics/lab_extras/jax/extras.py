import jax.numpy as jnp
import lab as B
from beartype.typing import List
from lab import dispatch
from plum import Union

_Numeric = Union[B.Number, B.JAXNumeric]


@dispatch
def polyval(coeffs: list, x: _Numeric) -> _Numeric:  # type: ignore
    """
    Computes the elementwise value of a polynomial.

    If `x` is a tensor and `coeffs` is a list if size n + 1, this function returns
    the value of the n-th order polynomial

    ..math:
        p(x) = coeffs[n-1] + coeffs[n-2] * x + ... + coeffs[0] * x**(n-1)
    """
    return jnp.polyval(jnp.r_[coeffs], x)


@dispatch
def from_numpy(a: B.JAXNumeric, b: Union[list, List, B.NPNumeric, B.Number, B.JAXNumeric]):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    return jnp.array(b)
