import random

import numpy as np
from shapely.affinity import scale, rotate, translate
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from data_utils.gaussian_model import GaussianModel
from data_utils.utils import minkowski_sum, find_random_point_on_contour, z_rotation_matrix, \
    compute_aabb_diagonal


def place_polygon(added_polygons, current_polygon):
    if len(added_polygons) == 0:
        added_polygons.append(current_polygon)
        return 0, 0

    current_polygon_inverted = scale(current_polygon, -1, -1, origin=Point(0, 0))

    impossible_placement = unary_union([minkowski_sum(ap, current_polygon_inverted) for ap in added_polygons]).convex_hull
    impossible_placement = impossible_placement.buffer(0.02)

    contour = impossible_placement.exterior
    random_point = find_random_point_on_contour(contour)

    translation_offset = (random_point.x, random_point.y)
    translated_polygon = translate(current_polygon, xoff=translation_offset[0], yoff=translation_offset[1])

    added_polygons.append(translated_polygon)

    return translation_offset


def compose_scene(models):
    result_model = GaussianModel.empty()

    added_polygons = []

    for model in models:
        aabb = model.get_aabb
        xy_aabb = aabb[:, :2]

        diagonal = compute_aabb_diagonal(aabb)
        scale_factor = 1 / diagonal
        scale_vector = np.repeat(scale_factor, 3)

        z_rotation = random.uniform(0, 360)

        xy_polygon = Polygon([
            (xy_aabb[0, 0], xy_aabb[0, 1]),
            (xy_aabb[0, 0], xy_aabb[1, 1]),
            (xy_aabb[1, 0], xy_aabb[1, 1]),
            (xy_aabb[1, 0], xy_aabb[0, 1])
        ])
        xy_polygon = scale(xy_polygon, scale_factor, scale_factor, origin=Point(0, 0))
        xy_polygon = rotate(xy_polygon, z_rotation, origin=Point(0, 0))
        xy_translation = place_polygon(added_polygons, xy_polygon)
        translation = np.array([xy_translation[0], xy_translation[1], 0])

        transformed_model = model.scaled(scale_vector).rotated(z_rotation_matrix(z_rotation)).translated(translation)

        result_model += transformed_model

    result_model = result_model.shuffled()

    return result_model
