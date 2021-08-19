# Spherical Harmonics

This package implements spherical harmonics in d-dimensions in Python. The spherical harmonics are constructed using a fundamental set 

[Dai and Xu, 2013](https://arxiv.org/pdf/1304.2585.pdf)

Definition 3.1, fundamental system


## Example

### 3 Dimensional
```python
import numpy as np
from spherical_harmonics import SphericalHarmonics
from spherical_harmonics.utils import l2norm

dimension = 3
max_degree = 10
# Returns all the spherical harmonics of degree 3 up to degree 10.
Phi = SphericalHarmonics(dimension, max_degree)

x = np.random.randn(101, dimension)  # Create random points to evaluation Phi
x = x / l2norm(x)  # normalize vectors
out = Phi(x)  # Evaluate spherical harmonics at `x`

# In 3D there are (2 * degree + 1) spherical harmonics per degree,
# so in total we have 400 spherical harmonics of degree 20 and smaller.
num_harmonics = 0
for degree in range(max_degree):
    num_harmonics += 2 * degree + 1
assert num_harmonics == 100

assert out.numpy().shape == (101, num_harmonics)
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

assert out.numpy().shape == (101, num_harmonics)
```

---
**NOTE**

The fundamental system are precomputed and stored in `spherical_harmonics/fundamental_system` for dimensions up to 20. For each dimension we precomputed the first 1024 spherical harmonics. This means that for each degree we support a varying number of maximum degree.

---

## Installation

The package is not available on PyPi. The recommended way to install it is to clone it from GitHub and to run (ideally in a [virtual environment](https://docs.python.org/3/tutorial/venv.html) or [`poetry`](https://python-poetry.org/) shell)
```
pip install -r requirements.py
```
followed by
```
python setup.py develop
```
These commands add the package `spherical_harmonics` to your Python path.

We also recommend installing the dependencies to run the tests
```
pip install -r test_requirements.py
```

Checking a if the installation was successful can be done by running the test
```
make test
```

## Citation

If this package was helpful cite the foolow