#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: pycf/plot_utils.py
"""
Plotting functions for Ckq potentials and g-tensors.

Based on https://scipython.com/blog/visualizing-the-real-forms-of-the-spherical-harmonics/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# The following import configures Matplotlib for 3D plotting.
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm_y

plt.rc("text", usetex=True)


def makegrid(npoints):
    """Create angular and Cartesian grids for surface plotting.

    Parameters
    ----------
    npoints : int
        Number of theta samples. Phi uses ``2 * npoints`` samples.

    Returns
    -------
    theta : ndarray, shape (2*npoints, npoints)
        Polar angle grid in radians.
    phi : ndarray, shape (2*npoints, npoints)
        Azimuthal angle grid in radians.
    xyz : ndarray, shape (3, 2*npoints, npoints)
        Unit-sphere Cartesian coordinates ``[x, y, z]`` on the same grid.
    """
    # Grids of polar and azimuthal angles
    theta = np.linspace(0, np.pi, npoints)
    phi = np.linspace(0, 2 * np.pi, npoints * 2)
    # Create a 2-D meshgrid of (theta, phi) angles.
    theta, phi = np.meshgrid(theta, phi)
    # Calculate the Cartesian coordinates of each point in the mesh.
    # We take phi to be angle from x axis
    xyz = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return theta, phi, xyz


def plot_surface(
    ax,
    C,
    xyz,
    plot_title,
    color_option="phase",
    colormap_option="hsv",
    axis_option="off",
    colorbar_option="off",
    azimuth=-37.5 + 90,
    elevation=30,
    shade_option=True,
    append_max_to_title=True,
    limit_scale=1.0,
    axis_line_scale=1.0,
    axis_text_scale=1.1,
    axis_line_color="0.5",
    axis_line_width=1,
    show_axis_labels=True,
    axis_occlusion_with_surface=False,
    axis_line_zorder=None,
):
    """Plot a 3D surface from a complex potential grid.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D axis used for plotting.
    C : ndarray, shape (n_phi, n_theta)
        Complex potential values on the angular mesh.
    xyz : ndarray, shape (3, n_phi, n_theta)
        Cartesian coordinates returned by ``makegrid``.
    plot_title : str
        Title shown on the axis.

    Notes
    -----
    The plotted radius is ``abs(C)``, so ``Cx``, ``Cy``, ``Cz`` all have shape
    ``(n_phi, n_theta)``.
    """
    absC = np.abs(C)
    phaseC = np.angle(C)
    if color_option == "phase":
        color_variable = phaseC
    elif color_option == "abs":
        color_variable = absC
    else:
        raise ValueError("color_option must be 'phase' or 'abs'")
    Cx, Cy, Cz = absC * xyz
    scale = 1.0
    ax_lim = np.amax(absC)
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap_option))
    if color_option == "phase":
        cmap.set_clim(-np.pi, np.pi)
    elif color_option == "abs":
        cmap.set_clim(0, np.amax(absC))
    else:
        raise ValueError("color_option must be 'phase' or 'abs'")

    def _draw_axis_segment(axis_index, start_t, end_t):
        if np.isclose(start_t, end_t):
            return
        xyz_line = np.zeros((3, 2), dtype=float)
        xyz_line[axis_index, 0] = start_t
        xyz_line[axis_index, 1] = end_t
        ax.plot(
            xyz_line[0],
            xyz_line[1],
            xyz_line[2],
            c=axis_line_color,
            lw=axis_line_width,
            zorder=axis_line_zorder,
        )

    # Draw a set of x, y, z axes for reference.
    axis_extent = axis_line_scale * ax_lim

    if axis_occlusion_with_surface:
        # Draw axis back halves first and front halves last.
        az = np.deg2rad(azimuth)
        el = np.deg2rad(elevation)
        view_dir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

        back_segments = []
        front_segments = []
        for axis_index in range(3):
            front_is_positive = view_dir[axis_index] >= 0
            if front_is_positive:
                back_segments.append((axis_index, -axis_extent, 0.0))
                front_segments.append((axis_index, 0.0, axis_extent))
            else:
                back_segments.append((axis_index, 0.0, axis_extent))
                front_segments.append((axis_index, -axis_extent, 0.0))

        for axis_index, start_t, end_t in back_segments:
            _draw_axis_segment(axis_index, start_t, end_t)

        ax.plot_surface(
            Cx,
            Cy,
            Cz,
            facecolors=cmap.to_rgba(color_variable),
            rstride=1,
            cstride=1,
            linewidth=0,
            shade=shade_option,
        )

        for axis_index, start_t, end_t in front_segments:
            _draw_axis_segment(axis_index, start_t, end_t)
    else:
        ax.plot_surface(
            Cx,
            Cy,
            Cz,
            facecolors=cmap.to_rgba(color_variable),
            rstride=1,
            cstride=1,
            linewidth=0,
            shade=shade_option,
        )
        _draw_axis_segment(0, -axis_extent, axis_extent)
        _draw_axis_segment(1, -axis_extent, axis_extent)
        _draw_axis_segment(2, -axis_extent, axis_extent)
    # Set the Axes limits and title, turn off the Axes frame.
    if append_max_to_title:
        plot_title = plot_title + " : max = {:1.1f}".format(ax_lim)
    ax.set_title(plot_title)
    axis_limit = limit_scale * ax_lim
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)
    ax.axis(axis_option)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if show_axis_labels:
        ax.text(ax_lim * axis_text_scale, 0, 0, "X")
        ax.text(0, ax_lim * axis_text_scale, 0, "Y")
        ax.text(0, 0, ax_lim * axis_text_scale, "Z")
    # view used in old matlab code
    # ax.view_init(azim=-37.5+90+180, elev=30)
    # look roughly down 111
    ax.view_init(azim=azimuth, elev=elevation)
    if colorbar_option == "on":
        ax.figure.colorbar(cmap, ax=ax)


def parse_ckq_key(key):
    """Parse a key in Ckq format (for example C11, C20)."""
    if not isinstance(key, str) or not key.startswith("C"):
        raise ValueError("Coefficient keys must be strings in Ckq format, e.g. C11, C20")

    suffix = key[1:]
    if len(suffix) != 2 or not suffix.isdigit():
        raise ValueError(
            "Coefficient key '{}' is invalid; expected Ckq with single-digit k and q".format(key)
        )

    k = int(suffix[0])
    q = int(suffix[1])
    if q > k:
        raise ValueError("Coefficient key '{}' is invalid; q must satisfy 0 <= q <= k".format(key))

    return k, q


def validate_parameters(parameters):
    """Validate coefficient keys and return parsed (key, k, q) tuples."""
    parsed = []
    for key in parameters:
        k, q = parse_ckq_key(key)
        parsed.append((key, k, q))
    return parsed


def ckq(k, q, theta, phi):
    """Compute one spherical-tensor basis function C_kq on a mesh.

    Parameters
    ----------
    k : int
        Tensor rank.
    q : int
        Tensor component with ``0 <= q <= k``.
    theta : ndarray, shape (n_phi, n_theta)
        Polar angle mesh in radians.
    phi : ndarray, shape (n_phi, n_theta)
        Azimuthal angle mesh in radians.

    Returns
    -------
    C : ndarray, shape (n_phi, n_theta), complex
        Complex basis-function values on the input mesh.
    """
    C = np.sqrt((4 * np.pi) / (2 * k + 1)) * sph_harm_y(k, q, theta, phi)
    return C


def build_potential(parameters, theta, phi):
    """Build total complex potential from any mix of Ckq terms.

    Note that for q > 0, the -q component is implicitly included via conjugation.

    Parameters
    ----------
    parameters : dict[str, complex | float]
        Mapping of coefficient names (for example ``C11``, ``C20``) to values.
    theta : ndarray, shape (n_phi, n_theta)
        Polar angle mesh in radians.
    phi : ndarray, shape (n_phi, n_theta)
        Azimuthal angle mesh in radians.

    Returns
    -------
    C : ndarray, shape (n_phi, n_theta), complex
        Total potential obtained by summing all specified terms.
    """
    parsed = validate_parameters(parameters)
    C = np.zeros(theta.shape, dtype=complex)

    for key, k, q in parsed:
        Bkq = parameters[key]
        C_temp = ckq(k, q, theta, phi) * Bkq
        if q == 0:
            C = C + C_temp
        else:
            # For q > 0 we implicitly include the -q component via conjugation.
            C = C + C_temp + np.conjugate(C_temp)

    return C


def normalize_gtensor(gtensor):
    """Convert gtensor input to a (3, 3) array.

    Accepts either a full (3, 3) matrix or any 9-element vector-like input
    (for example shape (9,), (9,1), or (1,9)).
    """
    gtensor_arr = np.asarray(gtensor, dtype=float)
    if gtensor_arr.shape == (3, 3):
        return gtensor_arr
    if gtensor_arr.size == 9:
        return gtensor_arr.reshape(3, 3)
    raise ValueError("gtensor must be shape (3, 3) or contain 9 elements")


def build_gtensor_surface(gtensor, theta, phi):
    """Build a directional g-value surface from a 3x3 g-tensor.

    Parameters
    ----------
    gtensor : ndarray
        g-tensor as shape ``(3, 3)`` or any 9-element vector-like input.
    theta : ndarray, shape (n_phi, n_theta)
        Polar angle mesh in radians.
    phi : ndarray, shape (n_phi, n_theta)
        Azimuthal angle mesh in radians.

    Returns
    -------
    gvals : ndarray, shape (n_phi, n_theta)
        Directional g-values computed as ``norm([x, y, z] @ gtensor)``.
    """
    gtensor_mat = normalize_gtensor(gtensor)

    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    directions = np.stack((x, y, z), axis=-1)
    g_cart = directions @ gtensor_mat
    gvals = np.linalg.norm(g_cart, axis=-1)
    return gvals


def plot_potential(
    ax,
    parameters={"C10": 1},
    title="C10",
    color_option="phase",
    colormap_option="hsv",
    axis_option="off",
    colorbar_option="off",
    limit_scale=1.0,
):
    """Build and plot one potential surface from Ckq coefficients.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D axis used for plotting.
    parameters : dict[str, complex | float]
        Ckq coefficients used to build the potential.
    limit_scale : float
        Axis half-range scale relative to max radius. Smaller values make the
        surface appear larger in the plot window.
    """
    theta, phi, xyz = makegrid(36)
    ax.clear()
    # print(parameters)
    C = build_potential(parameters, theta, phi)
    plot_surface(
        ax,
        C,
        xyz,
        title,
        color_option=color_option,
        colormap_option=colormap_option,
        axis_option=axis_option,
        colorbar_option=colorbar_option,
        limit_scale=limit_scale,
        axis_line_scale=1.0,
        axis_text_scale=1.3,
        axis_line_zorder=1000,
    )


def plot_gtensor(
    ax,
    gtensor,
    title="g-tensor",
    axis_option="off",
    colorbar_option="on",
    azimuth=-37.5 + 90,
    elevation=30,
    limit_scale=1.0,
):
    """Build and plot a g-tensor surface using the jet colormap.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D axis used for plotting.
    gtensor : ndarray
        g-tensor as shape ``(3, 3)`` or any 9-element vector-like input.
    colorbar_option : str
        Use ``"on"`` to show colorbar or ``"off"`` to hide it.
    limit_scale : float
        Axis half-range scale relative to max radius. Lower values make the
        surface fill more of the figure.
    """
    ax.clear()
    gtensor_mat = normalize_gtensor(gtensor)
    theta, phi, xyz = makegrid(36)
    gvals = build_gtensor_surface(gtensor_mat, theta, phi)
    max_g = np.max(gvals)

    plot_surface(
        ax,
        gvals,
        xyz,
        title,
        color_option="abs",
        colormap_option="jet",
        axis_option=axis_option,
        colorbar_option=colorbar_option,
        azimuth=azimuth,
        elevation=elevation,
        shade_option=False,
        append_max_to_title=True,
        limit_scale=limit_scale,
        axis_line_scale=1.2,
        axis_text_scale=1.3,
        axis_line_color="0.5",
        axis_line_width=1,
        show_axis_labels=False,
        axis_occlusion_with_surface=False,
        axis_line_zorder=1000,
    )

    # Axis labels on +x/+y/+z.
    tscale = 1.3 * max_g
    ax.text(0, 0, tscale, "Z")
    ax.text(0, tscale, 0, "Y")
    ax.text(tscale, 0, 0, "X")

    _, eigvecs = np.linalg.eig(gtensor_mat)
    pscale = 1.2 * max_g
    for i in range(3):
        v = np.real(eigvecs[:, i])
        ax.plot(
            pscale * np.array([-v[0], v[0]]),
            pscale * np.array([-v[1], v[1]]),
            pscale * np.array([-v[2], v[2]]),
            c="0.5",
            lw=2,
            linestyle="-.",
            zorder=1000,
            alpha=0.9,
        )


""" Testing stuff that is run only if we execute the file, not import """
if __name__ == "__main__":

    fig = plt.figure(1, figsize=plt.figaspect(1.0))
    fig.clf()
    ax = fig.add_subplot(projection="3d")
    plot_gtensor(
        ax,
        gtensor=[
            [1, 1, 0.0],
            [1, 4, 0],
            [0.0, 0, 4],
        ],
        title="g-tensor",
        colorbar_option="off",
    )

    fig = plt.figure(2, figsize=plt.figaspect(1.0))
    fig.clf()
    ax = fig.add_subplot(projection="3d")
    plot_potential(
        ax,
        parameters={"C11": 0.1, "C20": 1},
        title="X",
        color_option="phase",
        colormap_option="hsv",
        axis_option="off",
        colorbar_option="off",
    )

    plt.show()
