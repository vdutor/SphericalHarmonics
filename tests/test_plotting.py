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

from spherical_harmonics import SphericalHarmonics
from spherical_harmonics.plotting import plotly_plot_spherical_function


@pytest.fixture
def spherical_function_to_plot():
    phi = SphericalHarmonics(3, 20)
    func = lambda x: phi(x)[:, -1]
    _ = func(np.random.randn(1, 3))
    return func


# FIXME: the use_mesh=True test is commented out because it relies on meshzoo
# which now requires a paid license. This could be fixed by using, e.g.,
# potpourri3d instead of meshzoo.
# @pytest.mark.parametrize("use_mesh", [True, False])
@pytest.mark.parametrize("use_mesh", [False])
@pytest.mark.parametrize("animate_steps", [0, 11])
def test_plotting(spherical_function_to_plot, use_mesh, animate_steps):
    _ = plotly_plot_spherical_function(
        spherical_function_to_plot,
        resolution=10,
        animate_steps=animate_steps,
        use_mesh=use_mesh,
    )
