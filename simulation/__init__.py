"""
Datasets package for crystal kinetics droplet system.

This package provides utilities for:
- Lysozyme crystal generation and augmentation
- Image augmentation
- Data preparation for detection and dimension prediction
"""

from simulation.lysozyme import (
    get_lysozyme_3d_coords,
    get_augmented_image,
    rotate,
    rotate_around_line,
    get_lysozome_point_indices_seen_in_xy_plane,
    find_outer_points,
    get_node_pos,
)

from simulation.augmentation import (
    aug_image,
    add_salt_and_pepper_noise,
)

__all__ = [
    # Lysozyme functions
    'get_lysozyme_3d_coords',
    'get_augmented_image',
    'rotate',
    'rotate_around_line',
    'get_lysozome_point_indices_seen_in_xy_plane',
    'find_outer_points',
    'get_node_pos',
    # Augmentation functions
    'aug_image',
    'add_salt_and_pepper_noise',
]

