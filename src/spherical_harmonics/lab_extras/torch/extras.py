import lab as B
import torch
from beartype.typing import List
from lab import dispatch
from plum import Union

_Numeric = Union[B.Number, B.TorchNumeric]


@dispatch
def polyval(coeffs: list, x: _Numeric) -> _Numeric:  # type: ignore
    """
    Computes the elementwise value of a polynomial.

    If `x` is a tensor and `coeffs` is a list if size n + 1, this function returns
    the value of the n-th order polynomial

    ..math:
        p(x) = coeffs[n-1] + coeffs[n-2] * x + ... + coeffs[0] * x**(n-1)
    """
    curVal = 0
    for i in range(len(coeffs) - 1):
        curVal = (curVal + coeffs[i]) * x

    return curVal + coeffs[-1]


@dispatch
def from_numpy(
    a: B.TorchNumeric, b: Union[List, B.Number, B.NPNumeric, B.TorchNumeric]
):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    return torch.tensor(b)
