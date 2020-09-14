"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import numpy.linalg
from scipy.ndimage import median_filter

from .constants import _EYE_RADIUS_DEFAULT
from .geometry.intersections import nearest_point_on_sphere_to_line
from .geometry.primitives import Circle, Ellipse, Line
from .geometry.projections import (
    project_line_into_image_plane,
    project_point_into_image_plane,
    unproject_ellipse,
)
from .geometry.utilities import normalize
from .observation import Observation, ObservationStorage
from .refraction import Refractionizer

logger = logging.getLogger(__name__)


class TwoSphereModel(object):
    def __init__(
        self,
        settings={"focal_length": 283.0, "resolution": (192, 192), "maxlen": 10000},
    ):
        self.settings = settings

        self.refractionizer = Refractionizer()

        self.sphere_center = np.asarray([0.0, 0.0, 35.0])
        self.corrected_sphere_center = self.refractionizer.correct_sphere_center(
            np.asarray([[*self.sphere_center]])
        )[0]

        self.observation_storage = ObservationStorage(maxlen=self.settings["maxlen"])

        self.debug_info = {
            "cost": -1.0,
            "residuals": [],
            "angles": [],
            "Dierkes_lines": [],
        }

    # OBSERVATION HANDLING
    def _extract_ellipse(self, pupil_datum):
        width, height = self.settings["resolution"]
        center = (
            +(pupil_datum["ellipse"]["center"][0] - width / 2),
            -(pupil_datum["ellipse"]["center"][1] - height / 2),
        )
        minor_axis = pupil_datum["ellipse"]["axes"][0] / 2.0
        major_axis = pupil_datum["ellipse"]["axes"][1] / 2.0
        angle = -(pupil_datum["ellipse"]["angle"] + 90.0) * np.pi / 180.0
        ellipse = Ellipse(center, minor_axis, major_axis, angle)
        return ellipse

    def add_to_observation_storage(self, pupil_datum):
        ellipse = self._extract_ellipse(pupil_datum)
        circle_3d_pair = unproject_ellipse(ellipse, self.settings["focal_length"])
        if circle_3d_pair:
            observation = Observation(
                ellipse,
                circle_3d_pair,
                pupil_datum["timestamp"],
                self.settings["focal_length"],
            )
            self.observation_storage.add_observation(observation)
            return observation
        else:
            return False

    def set_sphere_center(self, new_sphere_center):
        self.sphere_center = new_sphere_center
        self.corrected_sphere_center = self.refractionizer.correct_sphere_center(
            np.asarray([[*self.sphere_center]])
        )[0]

    @staticmethod
    def deep_sphere_estimate(
        observations: Sequence[Observation],
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        # https://silo.tips/download/least-squares-intersection-of-lines
        # https://www.researchgate.net/publication/333490770_A_fast_approach_to_refraction-aware_eye-model_fitting_and_gaze_prediction

        global prev_center, prev_disambiguation, prev_t
        try:
            prev_center
        except NameError:
            prev_center = None
            prev_disambiguation = None
            prev_t = None

        aux_2d = np.array([obs.aux_2d for obs in observations])
        aux_3d = np.array([obs.aux_3d for obs in observations])
        gaze_2d = np.array(
            [[*obs.gaze_2d.origin, *obs.gaze_2d.direction] for obs in observations]
        )

        debug_info = None
        cutoff = None

        if prev_center is not None:

            debug_info = dict()

            # Final Dierkes lines
            Dierkes_lines = [
                obs.get_Dierkes_line(idx)
                for obs, idx in zip(observations, prev_disambiguation)
            ]

            # Calculate residuals and cost
            a_minus_p = [line.origin - prev_center for line in Dierkes_lines]
            residuals = [
                v.T @ m[idx, :3, :3] @ v
                for v, m, idx in zip(a_minus_p, aux_3d, prev_disambiguation)
            ]
            debug_info["residuals"] = residuals
            debug_info["cost"] = np.mean(residuals)

            def normalize(vector):
                return vector / np.linalg.norm(vector)

            debug_info["directions"] = [
                normalize(obs.circle_3d_pair[idx].normal)[0:2]
                for obs, idx in zip(observations, prev_disambiguation)
            ]

            if debug_info["cost"] > 2:
                filtered_residuals = median_filter(residuals, 50)
                changes = np.gradient(filtered_residuals)
                max_change_idx = np.argmax(changes)

                cutoff = observations[max_change_idx].timestamp

                observations = [obs for obs in observations if obs.timestamp >= cutoff]
                aux_2d = np.array([obs.aux_2d for obs in observations])
                aux_3d = np.array([obs.aux_3d for obs in observations])
                gaze_2d = np.array(
                    [
                        [*obs.gaze_2d.origin, *obs.gaze_2d.direction]
                        for obs in observations
                    ]
                )

        ### ESTIMATE NEW MODEL
        # Estimate projected sphere center by nearest intersection of 2d gaze lines
        sum_aux_2d = aux_2d.sum(axis=0)
        projected_sphere_center = np.linalg.pinv(sum_aux_2d[:2, :2]) @ sum_aux_2d[:2, 2]

        # Disambiguate Dierkes lines
        # We want gaze_2d to points towards the sphere center. gaze_2d was collected
        # from Dierkes[0]. If it points into the correct direction, we know that
        # Dierkes[0] is the correct one to use, otherwise we need to use Dierkes[1]. We
        # can check that with the sign of the dot product.
        gaze_2d_origins = gaze_2d[:, :2]
        gaze_2d_directions = gaze_2d[:, 2:]
        gaze_2d_towards_center = gaze_2d_origins - projected_sphere_center

        dot_products = np.sum(gaze_2d_towards_center * gaze_2d_directions, axis=1)
        disambiguation_indices = np.where(dot_products < 0, 1, 0)

        observation_indices = np.arange(len(disambiguation_indices))
        aux_3d_disambiguated = aux_3d[observation_indices, disambiguation_indices, :, :]

        # Estimate sphere center by nearest intersection of Dierkes lines
        sum_aux_3d = aux_3d_disambiguated.sum(axis=0)
        sphere_center = np.linalg.pinv(sum_aux_3d[:3, :3]) @ sum_aux_3d[:3, 3]

        prev_center = sphere_center
        prev_disambiguation = disambiguation_indices
        prev_t = observations[-1].timestamp

        return sphere_center, cutoff, debug_info

    # GAZE PREDICTION
    def _extract_unproject_disambiguate(self, pupil_datum):
        ellipse = self._extract_ellipse(pupil_datum)
        circle_3d_pair = unproject_ellipse(ellipse, self.settings["focal_length"])
        if circle_3d_pair:
            circle_3d = self._disambiguate_circle_3d_pair(circle_3d_pair)
        else:
            circle_3d = Circle([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.0)
        return circle_3d

    def _disambiguate_circle_3d_pair(self, circle_3d_pair):
        circle_center_2d = project_point_into_image_plane(
            circle_3d_pair[0].center, self.settings["focal_length"]
        )
        circle_normal_2d = normalize(
            project_line_into_image_plane(
                Line(circle_3d_pair[0].center, circle_3d_pair[0].normal),
                self.settings["focal_length"],
            ).direction
        )
        sphere_center_2d = project_point_into_image_plane(
            self.sphere_center, self.settings["focal_length"]
        )

        if np.dot(circle_center_2d - sphere_center_2d, circle_normal_2d) >= 0:
            return circle_3d_pair[0]
        else:
            return circle_3d_pair[1]

    def predict_pupil_circle(
        self, input_, from_given_circle_3d_pair=False, use_unprojection=False
    ):
        if from_given_circle_3d_pair:
            circle_3d = self._disambiguate_circle_3d_pair(input_)
        else:
            circle_3d = self._extract_unproject_disambiguate(input_)
        if circle_3d:
            direction = normalize(circle_3d.center)
            nearest_point_on_sphere = nearest_point_on_sphere_to_line(
                self.sphere_center, _EYE_RADIUS_DEFAULT, [0.0, 0.0, 0.0], direction
            )
            if use_unprojection:
                gaze_vector = circle_3d.normal
            else:
                gaze_vector = normalize(nearest_point_on_sphere - self.sphere_center)
            radius = np.linalg.norm(nearest_point_on_sphere) / np.linalg.norm(
                circle_3d.center
            )
            pupil_circle = Circle(nearest_point_on_sphere, gaze_vector, radius)
        else:
            pupil_circle = Circle([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.0)
        return pupil_circle

    def apply_refraction_correction(self, pupil_circle):
        input_features = np.asarray(
            [[*self.sphere_center, *pupil_circle.normal, pupil_circle.radius]]
        )
        refraction_corrected_params = self.refractionizer.correct_pupil_circle(
            input_features
        )[0]

        refraction_corrected_gaze_vector = normalize(refraction_corrected_params[:3])
        refraction_corrected_radius = refraction_corrected_params[-1]
        refraction_corrected_pupil_center = (
            self.corrected_sphere_center
            + _EYE_RADIUS_DEFAULT * refraction_corrected_gaze_vector
        )

        refraction_corrected_pupil_circle = Circle(
            refraction_corrected_pupil_center,
            refraction_corrected_gaze_vector,
            refraction_corrected_radius,
        )

        return refraction_corrected_pupil_circle

    # UTILITY FUNCTIONS
    def reset(self):
        self.sphere_center = np.array([0.0, 0.0, 35.0])
        self.observation_storage = ObservationStorage(maxlen=self.settings["maxlen"])
        self.debug_info = {
            "cost": -1.0,
            "residuals": [],
            "angles": [],
            "dierkes_lines": [],
        }
