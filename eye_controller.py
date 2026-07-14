import time
from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np

from input_controller import GameInputController


class EyeController:
    """Calibra os olhos e aciona a mira com um piscar válido."""

    LEFT_EYE_POINTS = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE_POINTS = (362, 385, 387, 263, 373, 380)

    def __init__(
        self,
        calibration_frames: int = 45,
        threshold_factor: float = 0.72,
        minimum_closed_time: float = 0.06,
        maximum_closed_time: float = 0.80,
        blink_cooldown: float = 0.70,
        aim_hold_mode: bool = False,
    ) -> None:
        self.calibration_frames = calibration_frames
        self.threshold_factor = threshold_factor
        self.minimum_closed_time = minimum_closed_time
        self.maximum_closed_time = maximum_closed_time
        self.blink_cooldown = blink_cooldown
        self.aim_hold_mode = aim_hold_mode

        self.calibration_values = deque(maxlen=calibration_frames)
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
        self.aim_is_active = False

    @staticmethod
    def distance(point_a, point_b) -> float:
        a = np.array([point_a.x, point_a.y], dtype=float)
        b = np.array([point_b.x, point_b.y], dtype=float)
        return float(np.linalg.norm(a - b))

    def calculate_eye_aspect_ratio(
        self,
        landmarks: Sequence,
        eye_points: Tuple[int, int, int, int, int, int],
    ) -> float:
        outer, upper_outer, upper_inner, inner, lower_inner, lower_outer = eye_points
        vertical_1 = self.distance(landmarks[upper_outer], landmarks[lower_outer])
        vertical_2 = self.distance(landmarks[upper_inner], landmarks[lower_inner])
        horizontal = self.distance(landmarks[outer], landmarks[inner])

        if horizontal == 0:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def calculate_eye_ratios(self, landmarks: Sequence) -> Tuple[float, float, float]:
        left = self.calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE_POINTS)
        right = self.calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE_POINTS)
        average = (left + right) / 2.0

        self.last_left_ratio = left
        self.last_right_ratio = right
        self.last_average_ratio = average

        return left, right, average

    def calibrate(self, average_ratio: float) -> None:
        if average_ratio <= 0:
            return

        self.calibration_values.append(average_ratio)
        if len(self.calibration_values) < self.calibration_frames:
            return

        self.normal_eye_ratio = float(np.median(np.array(self.calibration_values)))
        self.closed_threshold = self.normal_eye_ratio * self.threshold_factor
        self.open_threshold = self.closed_threshold * 1.12
        self.is_calibrated = True

    def reset(self) -> None:
        self.eyes_were_closed = False
        self.closed_at = 0.0

    def release_buttons(self) -> None:
        if self.aim_is_active:
            GameInputController.right_button_up()
            self.aim_is_active = False

    def reset_calibration(self) -> None:
        self.release_buttons()
        self.reset()
        self.calibration_values.clear()
        self.is_calibrated = False
        self.normal_eye_ratio = 0.0
        self.closed_threshold = 0.0
        self.open_threshold = 0.0

    def toggle_aim(self) -> str:
        if self.aim_hold_mode:
            if self.aim_is_active:
                GameInputController.right_button_up()
                self.aim_is_active = False
                return "MIRA DESATIVADA"

            GameInputController.right_button_down()
            self.aim_is_active = True
            return "MIRA ATIVADA"

        GameInputController.right_click(0.08)
        return "MIRA"

    def process(self, landmarks: Sequence) -> Optional[str]:
        now = time.monotonic()
        left, right, average = self.calculate_eye_ratios(landmarks)

        if not self.is_calibrated:
            self.calibrate(average)
            remaining = max(self.calibration_frames - len(self.calibration_values), 0)
            return f"CALIBRANDO OLHOS: {remaining}"

        both_closed = left < self.closed_threshold and right < self.closed_threshold
        both_open = left > self.open_threshold and right > self.open_threshold

        if both_closed and not self.eyes_were_closed:
            self.eyes_were_closed = True
            self.closed_at = now
            return None

        if both_open and self.eyes_were_closed:
            closed_duration = now - self.closed_at
            self.eyes_were_closed = False

            valid_blink = (
                self.minimum_closed_time <= closed_duration <= self.maximum_closed_time
                and now - self.last_blink_at >= self.blink_cooldown
            )

            if valid_blink:
                self.last_blink_at = now
                action = self.toggle_aim()
                print(action)
                return action

        if self.eyes_were_closed and now - self.closed_at > self.maximum_closed_time:
            self.reset()

        return None

    def get_debug_text(self) -> str:
        if not self.is_calibrated:
            return f"Calibracao: {len(self.calibration_values)}/{self.calibration_frames}"

        aim = "ON" if self.aim_is_active else "OFF"
        return (
            f"L:{self.last_left_ratio:.3f} R:{self.last_right_ratio:.3f} "
            f"AVG:{self.last_average_ratio:.3f} LIM:{self.closed_threshold:.3f} "
            f"MIRA:{aim}"
        )