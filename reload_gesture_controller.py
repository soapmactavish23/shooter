import time
from typing import Optional, Sequence

import numpy as np

from input_controller import GameInputController


class ReloadGestureController:
    """
    Recarrega quando duas mãos abertas ficam alinhadas verticalmente,
    com uma palma abaixo da outra, durante um curto período.
    """

    def __init__(
        self,
        confirmation_time: float = 0.32,
        cooldown: float = 1.25,
        max_horizontal_gap_ratio: float = 1.25,
        min_vertical_gap_ratio: float = 0.45,
        max_vertical_gap_ratio: float = 2.60,
        reload_key: str = "r",
        straight_angle_threshold: float = 145.0,
    ) -> None:
        self.confirmation_time = confirmation_time
        self.cooldown = cooldown
        self.max_horizontal_gap_ratio = max_horizontal_gap_ratio
        self.min_vertical_gap_ratio = min_vertical_gap_ratio
        self.max_vertical_gap_ratio = max_vertical_gap_ratio
        self.reload_key = reload_key
        self.straight_angle_threshold = straight_angle_threshold

        self.pose_started_at = 0.0
        self.pose_active = False
        self.reload_locked = False
        self.last_reload_at = 0.0

        self.last_horizontal_ratio = 0.0
        self.last_vertical_ratio = 0.0
        self.last_hands_open = False

    @staticmethod
    def point(landmark) -> np.ndarray:
        return np.array(
            [
                landmark.x,
                landmark.y,
                landmark.z,
            ],
            dtype=float,
        )

    @staticmethod
    def calculate_angle(
        point_a,
        point_b,
        point_c,
    ) -> float:
        vector_ba = (
            ReloadGestureController.point(point_a)
            - ReloadGestureController.point(point_b)
        )
        vector_bc = (
            ReloadGestureController.point(point_c)
            - ReloadGestureController.point(point_b)
        )

        denominator = (
            np.linalg.norm(vector_ba)
            * np.linalg.norm(vector_bc)
        )

        if denominator == 0:
            return 0.0

        cosine = np.clip(
            np.dot(vector_ba, vector_bc)
            / denominator,
            -1.0,
            1.0,
        )

        return float(
            np.degrees(
                np.arccos(cosine)
            )
        )

    def finger_is_extended(
        self,
        landmarks: Sequence,
        mcp_index: int,
        pip_index: int,
        tip_index: int,
    ) -> bool:
        return (
            self.calculate_angle(
                landmarks[mcp_index],
                landmarks[pip_index],
                landmarks[tip_index],
            )
            >= self.straight_angle_threshold
        )

    def hand_is_open(
        self,
        landmarks: Sequence,
    ) -> bool:
        extended_fingers = (
            self.finger_is_extended(
                landmarks,
                5,
                6,
                8,
            ),
            self.finger_is_extended(
                landmarks,
                9,
                10,
                12,
            ),
            self.finger_is_extended(
                landmarks,
                13,
                14,
                16,
            ),
            self.finger_is_extended(
                landmarks,
                17,
                18,
                20,
            ),
        )

        # Exige pelo menos três dedos longos estendidos.
        return sum(extended_fingers) >= 3

    def palm_center(
        self,
        landmarks: Sequence,
    ) -> np.ndarray:
        indexes = (
            0,
            5,
            9,
            13,
            17,
        )

        return np.mean(
            [
                self.point(
                    landmarks[index]
                )[:2]
                for index in indexes
            ],
            axis=0,
        )

    def palm_width(
        self,
        landmarks: Sequence,
    ) -> float:
        return float(
            np.linalg.norm(
                self.point(
                    landmarks[5]
                )[:2]
                - self.point(
                    landmarks[17]
                )[:2]
            )
        )

    def is_reload_pose(
        self,
        left_landmarks: Sequence,
        right_landmarks: Sequence,
    ) -> bool:
        left_open = self.hand_is_open(
            left_landmarks,
        )
        right_open = self.hand_is_open(
            right_landmarks,
        )

        self.last_hands_open = (
            left_open and right_open
        )

        if not self.last_hands_open:
            return False

        left_center = self.palm_center(
            left_landmarks,
        )
        right_center = self.palm_center(
            right_landmarks,
        )

        horizontal_gap = abs(
            float(
                left_center[0]
                - right_center[0]
            )
        )
        vertical_gap = abs(
            float(
                left_center[1]
                - right_center[1]
            )
        )

        average_palm_width = max(
            (
                self.palm_width(
                    left_landmarks,
                )
                + self.palm_width(
                    right_landmarks,
                )
            )
            / 2.0,
            0.001,
        )

        self.last_horizontal_ratio = (
            horizontal_gap
            / average_palm_width
        )
        self.last_vertical_ratio = (
            vertical_gap
            / average_palm_width
        )

        return (
            self.last_horizontal_ratio
            <= self.max_horizontal_gap_ratio
            and self.min_vertical_gap_ratio
            <= self.last_vertical_ratio
            <= self.max_vertical_gap_ratio
        )

    def process(
        self,
        left_landmarks: Sequence,
        right_landmarks: Sequence,
    ) -> Optional[str]:
        now = time.monotonic()

        reload_pose = self.is_reload_pose(
            left_landmarks,
            right_landmarks,
        )

        if not reload_pose:
            self.pose_started_at = 0.0
            self.pose_active = False
            self.reload_locked = False
            return None

        if self.reload_locked:
            return None

        if not self.pose_active:
            self.pose_active = True
            self.pose_started_at = now
            return "MANTENHA UMA MAO ABAIXO DA OUTRA"

        confirmed = (
            now - self.pose_started_at
            >= self.confirmation_time
        )
        cooldown_finished = (
            now - self.last_reload_at
            >= self.cooldown
        )

        if confirmed and cooldown_finished:
            GameInputController.press_key(
                self.reload_key,
            )

            self.last_reload_at = now
            self.reload_locked = True

            print(
                "Reload enviado | "
                f"X={self.last_horizontal_ratio:.2f} | "
                f"Y={self.last_vertical_ratio:.2f}"
            )

            return "RECARREGANDO"

        return "MANTENHA UMA MAO ABAIXO DA OUTRA"

    def reset_tracking(self) -> None:
        self.pose_started_at = 0.0
        self.pose_active = False
        self.reload_locked = False

    def get_debug_text(self) -> str:
        open_text = (
            "SIM"
            if self.last_hands_open
            else "NAO"
        )

        return (
            f"RELOAD ABERTAS:{open_text} | "
            f"X:{self.last_horizontal_ratio:.2f} | "
            f"Y:{self.last_vertical_ratio:.2f}"
        )