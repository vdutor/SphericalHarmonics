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

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from beartype.typing import Callable
from matplotlib import cm


def spherical_grid(resolution=100):
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.vstack([x.flatten(), y.flatten(), z.flatten()]).T


def plot_spherical_function(
    f: Callable, resolution: int = 100, rescale_radius: bool = False, ax=None
):
    """
    Use matplotlib to create a 3D plot of a function defined on the sphere.

    :param f: A function which takes a [N, 3] matrix of points on the sphere in
        cartesian coordinates, and returns a sized [N] vector.
    :param resolution: Controls the resolution of the grid (larger means finer grid).
    :param rescale_radius: Slightly modify the radius to depict the function values.
    :param ax: Matplotlib axis to plot. If `None` (default), a new axis is created.
    """

    grid = spherical_grid(resolution)
    fgrid = f(grid).reshape(resolution, resolution)

    # Set the aspect ratio to 1 so our sphere looks spherical
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if rescale_radius:
        scale = np.abs(fgrid)
    else:
        scale = 1.0

    # scale the colors
    fmax, fmin = fgrid.max(), fgrid.min()
    fcolors = (fgrid - fmin) / (fmax - fmin)
    viridis = cm.get_cmap("viridis")

    ax.plot_surface(
        grid[:, 0].reshape(resolution, resolution) * scale,
        grid[:, 1].reshape(resolution, resolution) * scale,
        grid[:, 2].reshape(resolution, resolution) * scale,
        rstride=1,
        cstride=1,
        facecolors=viridis(fcolors),
    )

    # Turn off the axis planes
    ax.set_axis_off()
    return ax


def get_mesh_on_sphere(resolution):
    import meshzoo

    grid, cells = meshzoo.icosa_sphere(resolution)
    return grid, cells


def plotly_plot_spherical_function(
    f: Callable, use_mesh: bool = True, resolution: int = 100, animate_steps: int = 0
):
    """
    Use plotly to create a 3D plot of a function defined on the sphere.

    :param f: A function which takes a [N, 3] matrix of points on the sphere in
        cartesian coordinates, and returns a sized [N] vector.
    :param use_mesh: If `True` use a mesh for plotting, otherwise a Surface plot is used.
    :param resolution: Controls the resolution of the mesh (larger means finer mesh).
    :param animate_steps: An animation, in which the sphere is rotating
        `animate_steps` number of steps can be created if setting `animate_steps
        > 0`. If set to 0 (default), no animation is added to the plot.
    """

    if use_mesh:
        grid, cells = get_mesh_on_sphere(resolution)
        x, y, z = [grid[:, i] for i in range(3)]
        fgrid = f(grid)
    else:
        phi = np.linspace(0, np.pi, resolution)
        theta = np.linspace(0, 2 * np.pi, resolution)
        phi, theta = np.meshgrid(phi, theta)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        grid = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
        fgrid = f(grid).reshape(resolution, resolution)

    # scale the colors
    # fmax, fmin = fgrid.max(), fgrid.min()
    # fcolors = (fgrid - fmin) / (fmax - fmin)
    fcolors = fgrid

    C0 = "rgb(31, 119, 180)"
    C1 = "rgb(255, 127, 14)"
    colorscale = [[0.0, C0], [0.5, "white"], [1, C1]]
    if use_mesh:
        plot = go.Mesh3d(
            x=grid[:, 0],
            y=grid[:, 1],
            z=grid[:, 2],
            i=cells[:, 0],
            j=cells[:, 1],
            k=cells[:, 2],
            colorbar_title="z",
            colorscale=colorscale,
            intensity=fcolors,
        )
    else:
        plot = go.Surface(x=x, y=y, z=z, surfacecolor=fcolors, colorscale=colorscale)
    fig = go.Figure(plot)
    fig.update_layout(scene_aspectmode="cube")
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    # fig.update_traces(showscale=False, hoverinfo="none")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False),
        )
    )

    x_eye = 1
    y_eye = 1
    z_eye = 1
    angle = np.pi / 720

    fig.update_layout(
        width=600,
        height=600,
        scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
    )

    frames = []

    def rotate(x, y, z):
        r, t, z = xyz2rtz(x, y, z)
        t += angle
        x, y, z = rtz2xyz(r, t, z)
        frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=x, y=y, z=z))))
        return x, y, z

    def xyz2rtz(x, y, z):
        return (x**2 + y**2) ** 0.5, np.arctan2(y, x), z

    def rtz2xyz(r, t, z):
        return r * np.cos(t), r * np.sin(t), z

    x, y, z = x_eye, y_eye, z_eye

    if animate_steps > 0:
        for i in range(animate_steps):
            print(i)
            x, y, z = rotate(x, y, z)
        fig.frames = frames

    return fig
