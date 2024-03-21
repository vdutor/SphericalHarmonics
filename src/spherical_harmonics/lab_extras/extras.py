import lab as B
from beartype.typing import List
from lab import dispatch
from lab.util import abstract
from plum import Union


@dispatch
@abstract()
def polyval(coeffs: list, x: B.Numeric):
    """
    Computes the elementwise value of a polynomial.

    If `x` is a tensor and `coeffs` is a list if size n + 1, this function returns
    the value of the n-th order polynomial

    ..math:
        p(x) = coeffs[n-1] + coeffs[n-2] * x + ... + coeffs[0] * x**(n-1)
    """


@dispatch
@abstract()
def from_numpy(_: B.Numeric, b: Union[list, List, B.Numeric, B.NPNumeric]):
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
