# Spherical Harmonics

This package implements spherical harmonics in d-dimensions in Python. The spherical harmonics are defined as zonal functions through the Gegenbauer polynomials and a fundamental system of points (see [Dai and Xu (2013)](https://arxiv.org/pdf/1304.2585.pdf), defintion 3.1). The spherical harmonics form a ortho-normal set on the hypersphere. This package implements a greedy algorithm to compute the fundamental set for dimensions up to 20.

The computations of this package can be carried out in either TensorFlow, Pytorch, Jax or NumPy.
A specific backend can be chosen by simply importing it as follows
```
import spherical_harmonics.tensorflow  # noqa
```

## Example

### 3 Dimensional
```python
import tensorflow as tf
import spherical_harmonics.tensorflow  # run computation in TensorFlow

from spherical_harmonics import SphericalHarmonics
from spherical_harmonics.utils import l2norm

dimension = 3
max_degree = 10
# Returns all the spherical harmonics in dimension 3 up to degree 10.
Phi = SphericalHarmonics(dimension, max_degree)

x = tf.random.normal((101, dimension))  # Create random points to evaluate Phi
x = x / tf.norm(x, axis=1, keepdims=True)  # Normalize vectors
out = Phi(x)  # Evaluate spherical harmonics at `x`

# In 3D there are (2 * degree + 1) spherical harmonics per degree,
# so in total we have 400 spherical harmonics of degree 20 and smaller.
num_harmonics = 0
for degree in range(max_degree):
    num_harmonics += 2 * degree + 1
assert num_harmonics == 100

assert out.shape == (101, num_harmonics)
```

### 4 Dimensional

The setup for 4 dimensional spherical harmonics is very similar to the 3D case. Note that there are more spherical harmonics now of degree smaller than 20.

```python
import numpy as np
from spherical_harmonics import SphericalHarmonics
from spherical_harmonics.utils import l2norm

dimension = 4
max_degree = 10
# Returns all the spherical harmonics of degree 4 up to degree 10.
Phi = SphericalHarmonics(dimension, max_degree)

x = np.random.randn(101, dimension)  # Create random points to evaluation Phi
x = x / l2norm(x)  # normalize vectors
out = Phi(x)  # Evaluate spherical harmonics at `x`

# In 4D there are (degree + 1)**2 spherical harmonics per degree,
# so in total we have 385 spherical harmonics of degree 20 and smaller.
num_harmonics = 0
for degree in range(max_degree):
    num_harmonics += (degree + 1) ** 2
assert num_harmonics == 385

assert out.shape == (101, num_harmonics)
```

---
**NOTE**

The fundamental systems up to dimensions 20 are precomputed and stored in `spherical_harmonics/fundamental_system`. For each dimension we precompute the first amount of spherical harmonics. This means that in each dimension we support a varying number of maximum degree (`max_degree`) and number of spherical harmonics:

| Dimension | Max Degree | Number Harmonics |
|----------:|-----------:|-----------------:|
|         3 |         34 |             1156 |
|         4 |         20 |             2870 |
|         5 |         10 |            16170 |
|         6 |          8 |             1254 |
|         7 |          7 |             1386 |
|         8 |          6 |             1122 |
|         9 |          6 |             1782 |
|        10 |          6 |             2717 |
|        11 |          5 |             1287 |
|        12 |          5 |             1729 |
|        13 |          5 |             2275 |
|        14 |          5 |             2940 |
|        15 |          5 |             3740 |
|        16 |          4 |              952 |
|        17 |          4 |             1122 |
|        18 |          4 |             1311 |
|        19 |          4 |             1520 |
|        20 |          4 |             1750 |

To precompute a larger fundamental system for a dimension run the following script
```
cd spherical_harmonics
python fundament_set.py
```
after specifying the desired options in the file.

---

## Installation

The package is now available on PyPI under the name of `spherical-harmonics-basis`.

Simply run
```
pip install spherical-harmonics-basis
```


## Citation

If this code was useful for your research, please consider citing the following [paper](http://proceedings.mlr.press/v119/dutordoir20a/dutordoir20a.pdf):
```
@inproceedings{Dutordoir2020spherical,
  title     = {{Sparse Gaussian Processes with Spherical Harmonic Features}},
  author    = {Dutordoir, Vincent and Durrande, Nicolas and Hensman, James},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  date      = {2020},
}
```
