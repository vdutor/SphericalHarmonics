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

from setuptools import find_packages, setup

requirements = [
    "numpy",
    "scipy",
]

setup(
    name="Spherical Harmonics",
    version="0.0.2a0",
    author="vd309@cam.ac.uk",
    description="Python Implementation of Spherical harmonics in dimension >= 3",
    packages=find_packages("src"),  # include all packages under src
    package_dir={"": "src"},  # tell distutils packages are under src
    install_requires=requirements,
    include_package_data=True,
    package_data={"spherical_harmonics": ["fundamental_system/*.npz"]},
)
