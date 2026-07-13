import time
from collections import deque
from typing import Optional, Sequence, Tuple
import pyautogui
import numpy as np


class EyeController:
    """
    Detects blinks and winks using MediaPipe Face Mesh.

    The controller automatically calibrates the normal eye opening
    before enabling blink detection.r
    """

    LEFT_EYE_POINTS = (
        33,
        160,
        158,
        133,
        153,
        144,
    )

    RIGHT_EYE_POINTS = (
        362,
        385,
        387,
        263,
        373,
        380,
    )

    def __init__(
        self,
        calibration_frames: int = 45,
        threshold_factor: float = 0.72,
        minimum_closed_time: float = 0.05,
        maximum_closed_time: float = 1.20,
        blink_cooldown: float = 0.60,
    ) -> None:
        self.calibration_frames = calibration_frames
        self.threshold_factor = threshold_factor

        self.minimum_closed_time = minimum_closed_time
        self.maximum_closed_time = maximum_closed_time
        self.blink_cooldown = blink_cooldown

        self.calibration_values = deque(
            maxlen=calibration_frames,
        )

        self.is_calibrated = False

        self.normal_eye_ratio = 0.0
        self.closed_threshold = 0.0
        self.open_threshold = 0.0

        self.eyes_were_closed = False
        self.closed_at = 0.0
        self.last_blink_at = 0.0

        self.last_left_ratio = 0.0
        self.last_right_ratio = 0.0
        self.last_average_ratio = 0.0

    @staticmethod
    def distance(
        point_a,
        point_b,
    ) -> float:
        point_a_array = np.array(
            [
                point_a.x,
                point_a.y,
            ],
            dtype=float,
        )

        point_b_array = np.array(
            [
                point_b.x,
                point_b.y,
            ],
            dtype=float,
        )

        return float(
            np.linalg.norm(
                point_a_array - point_b_array
            )
        )

    def calculate_eye_aspect_ratio(
        self,
        landmarks: Sequence,
        eye_points: Tuple[
            int,
            int,
            int,
            int,
            int,
            int,
        ],
    ) -> float:
        (
            outer_corner,
            upper_outer,
            upper_inner,
            inner_corner,
            lower_inner,
            lower_outer,
        ) = eye_points

        vertical_distance_1 = self.distance(
            landmarks[upper_outer],
            landmarks[lower_outer],
        )

        vertical_distance_2 = self.distance(
            landmarks[upper_inner],
            landmarks[lower_inner],
        )

        horizontal_distance = self.distance(
            landmarks[outer_corner],
            landmarks[inner_corner],
        )

        if horizontal_distance == 0:
            return 0.0

        return (
            vertical_distance_1
            + vertical_distance_2
        ) / (
            2.0
            * horizontal_distance
        )

    def calculate_eye_ratios(
        self,
        landmarks: Sequence,
    ) -> Tuple[float, float, float]:
        left_ratio = self.calculate_eye_aspect_ratio(
            landmarks,
            self.LEFT_EYE_POINTS,
        )

        right_ratio = self.calculate_eye_aspect_ratio(
            landmarks,
            self.RIGHT_EYE_POINTS,
        )

        average_ratio = (
            left_ratio + right_ratio
        ) / 2.0

        self.last_left_ratio = left_ratio
        self.last_right_ratio = right_ratio
        self.last_average_ratio = average_ratio

        return (
            left_ratio,
            right_ratio,
            average_ratio,
        )

    def calibrate(
        self,
        average_ratio: float,
    ) -> None:
        """
        Collects normal open-eye values.

        During calibration, keep your eyes open and look toward
        the webcam.
        """
        if average_ratio <= 0:
            return

        self.calibration_values.append(
            average_ratio,
        )

        if (
            len(self.calibration_values)
            < self.calibration_frames
        ):
            return

        values = np.array(
            self.calibration_values,
            dtype=float,
        )

        self.normal_eye_ratio = float(
            np.median(values)
        )

        self.closed_threshold = (
            self.normal_eye_ratio
            * self.threshold_factor
        )

        self.open_threshold = (
            self.closed_threshold
            * 1.12
        )

        self.is_calibrated = True

        print(
            "Eye calibration completed | "
            f"Open: {self.normal_eye_ratio:.3f} | "
            f"Closed threshold: "
            f"{self.closed_threshold:.3f}"
        )

    def reset(self) -> None:
        self.eyes_were_closed = False
        self.closed_at = 0.0

    def reset_calibration(self) -> None:
        self.reset()

        self.calibration_values.clear()
        self.is_calibrated = False

        self.normal_eye_ratio = 0.0
        self.closed_threshold = 0.0
        self.open_threshold = 0.0

    def process(
        self,
        landmarks: Sequence,
    ) -> Optional[str]:
        now = time.time()

        (
            left_ratio,
            right_ratio,
            average_ratio,
        ) = self.calculate_eye_ratios(
            landmarks,
        )

        if not self.is_calibrated:
            self.calibrate(
                average_ratio,
            )

            remaining_frames = (
                self.calibration_frames
                - len(self.calibration_values)
            )

            return (
                f"CALIBRATING EYES: "
                f"{max(remaining_frames, 0)}"
            )

        left_eye_closed = (
            left_ratio
            < self.closed_threshold
        )

        right_eye_closed = (
            right_ratio
            < self.closed_threshold
        )

        eyes_closed_now = (
            left_eye_closed
            or right_eye_closed
        )

        eyes_open_now = (
            left_ratio
            > self.open_threshold
            and right_ratio
            > self.open_threshold
        )

        if (
            eyes_closed_now
            and not self.eyes_were_closed
        ):
            self.eyes_were_closed = True
            self.closed_at = now
            pyautogui.click(button="right")
            return "EYES CLOSED"

        if (
            eyes_open_now
            and self.eyes_were_closed
        ):
            closed_duration = (
                now - self.closed_at
            )

            self.eyes_were_closed = False

            valid_blink = (
                self.minimum_closed_time
                <= closed_duration
                <= self.maximum_closed_time
                and now - self.last_blink_at
                >= self.blink_cooldown
            )

            if valid_blink:
                print("Aim")

                self.last_blink_at = now

                return "AIM"

        if (
            self.eyes_were_closed
            and now - self.closed_at
            > self.maximum_closed_time
        ):
            self.reset()

        return None

    def get_debug_text(self) -> str:
        if not self.is_calibrated:
            return (
                f"Eye calibration: "
                f"{len(self.calibration_values)}"
                f"/{self.calibration_frames}"
            )

        return (
            f"L: {self.last_left_ratio:.3f} | "
            f"R: {self.last_right_ratio:.3f} | "
            f"AVG: {self.last_average_ratio:.3f} | "
            f"LIMIT: {self.closed_threshold:.3f}"
        )
