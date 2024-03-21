import lab as B
import tensorflow as tf
from beartype.typing import List
from lab import dispatch
from plum import Union

_Numeric = Union[B.Number, B.TFNumeric, B.NPNumeric]


@dispatch
def polyval(coeffs: list, x: _Numeric) -> _Numeric:  # type: ignore
    """
    Computes the elementwise value of a polynomial.

    If `x` is a tensor and `coeffs` is a list if size n + 1, this function returns
    the value of the n-th order polynomial

    ..math:
        p(x) = coeffs[n-1] + coeffs[n-2] * x + ... + coeffs[0] * x**(n-1)
    """
    coeff_list = [B.cast(B.dtype(x), coeff) for coeff in coeffs]
    return tf.math.polyval(coeff_list, x)


@dispatch
def from_numpy(a: B.TFNumeric, b: Union[list, List, B.Numeric, B.NPNumeric, B.TFNumeric]):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    return tf.convert_to_tensor(b)
