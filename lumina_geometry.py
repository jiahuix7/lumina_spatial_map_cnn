"""Survey geometry, cosmology and magnitude helpers used by spatial_maps.ipynb.

Copied from compute_UVLF_cv.py (my SURF 2026 Lumina cosmic variance code) so
this project runs on its own. Only the pieces the notebook imports are here.
"""
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as con

# Lumina Planck cosmology parameters (Lumina methods paper, Table 2)
lumina_cosmology = FlatLambdaCDM(
    H0=67.66 * u.km / u.s / u.Mpc,
    Om0=0.3096,
    Ob0=4.897 / 100,
    Tcmb0=2.7255 * u.K,
    Neff=3.046,
    m_nu=0.0 * u.eV,
)

# Survey design: JWST NIRCam-like field at z ~ 10
z_min = 9.2
z_max = 10.9
z_center = 0.5 * (z_min + z_max)
nircam_width = 4.4 * u.arcmin
nircam_height = 2.2 * u.arcmin

MUV_limits = np.array([-19.5, -20.0, -20.5])
best_delta_MUV = 1  # A1: every galaxy brightened by 1 mag
best_sigma_UV = 1   # C1: 1 mag of scatter in M_UV at fixed halo
random_seed = 42


def fuv_to_MUV(fuv):
    """Convert L_lambda at 1500 A (erg/s/A) to absolute AB magnitude M_UV."""
    distance_cm = 10 * con.pc.cgs.value
    area_10pc = 4 * np.pi * distance_cm**2
    f_lambda = fuv / area_10pc
    f_lambda_per_cm = f_lambda * 1e8
    lambda_cm = 1500e-8
    c_cm_s = con.c.cgs.value
    f_nu = f_lambda_per_cm * lambda_cm**2 / c_cm_s
    return -2.5 * np.log10(f_nu) - 48.6


def calculate_rectangle_field_lengths(cosmology, z_center, z_min, z_max, width_angle, height_angle):
    """Comoving (width, height, line-of-sight depth) of a rectangular field, in cMpc."""
    transverse_distance = cosmology.comoving_transverse_distance(z_center).to_value(u.Mpc)
    los_depth = (cosmology.comoving_distance(z_max) - cosmology.comoving_distance(z_min)).to_value(u.Mpc)

    width_cmpc = transverse_distance * width_angle.to_value(u.rad)
    height_cmpc = transverse_distance * height_angle.to_value(u.rad)

    field_lengths_cmpc = np.array([width_cmpc, height_cmpc, los_depth])

    print("rectangle field lengths [cMpc]:", field_lengths_cmpc)
    print("rectangle field volume [cMpc^3]:", np.prod(field_lengths_cmpc))

    return field_lengths_cmpc
