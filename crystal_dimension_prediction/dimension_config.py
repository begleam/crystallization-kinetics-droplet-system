"""
Configuration file for crystal dimension prediction.
Contains parameter ranges and constants used across training and inference.
"""
import numpy as np

# Parameter ranges for crystal dimensions
w_min, w_max = 2, 282
l_min, l_max = 2, 282
theta_min, theta_max = 0 / 180 * np.pi, 90 / 180 * np.pi
phi_min, phi_max = 0 / 180 * np.pi, 360 / 180 * np.pi
gamma_min, gamma_max = 0 / 180 * np.pi, 90 / 180 * np.pi

# Alpha constant for crystal shape
ALPHA = 0.326297

