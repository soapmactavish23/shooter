import time
from typing import Dict, Optional, Sequence

import numpy as np
import pyautogui


class RightHandController:
    """
    Handles the physical right hand.

    Responsibilities:
    - Detect the gun-ready pose.
    - Detect the trigger-pull movement.
    - Perform a left mouse click.
    """

    def __init__(
        self,
        straight_angle_threshold: float = 155.0,
        thumb_angle_threshold: float = 145.0,
        trigger_timeout: float = 2.0,
        trigger_cooldown: float = 0.50,
    ) -> None:
        self.straight_angle_threshold = straight_angle_threshold
        self.thumb_angle_threshold = thumb_angle_threshold
        self.trigger_timeout = trigger_timeout
        self.trigger_cooldown = trigger_cooldown

        self.gun_is_ready = False
        self.trigger_is_locked = False

        self.gun_ready_at = 0.0
        self.last_trigger_at = 0.0

    @staticmethod
    def calculate_angle(point_a, point_b, point_c) -> float:
        vector_ba = np.array(
            [
                point_a.x - point_b.x,
                point_a.y - point_b.y,
                point_a.z - point_b.z,
            ],
            dtype=float,
        )

        vector_bc = np.array(
            [
                point_c.x - point_b.x,
                point_c.y - point_b.y,
                point_c.z - point_b.z,
            ],
            dtype=float,
        )

        denominator = np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)

        if denominator == 0:
            return 0.0

        cosine = np.dot(vector_ba, vector_bc) / denominator
        cosine = np.clip(cosine, -1.0, 1.0)

        return float(np.degrees(np.arccos(cosine)))

    def is_finger_extended(
        self,
        landmarks: Sequence,
        mcp_index: int,
        pip_index: int,
        tip_index: int,
    ) -> bool:
        angle = self.calculate_angle(
            landmarks[mcp_index],
            landmarks[pip_index],
            landmarks[tip_index],
        )

        return angle >= self.straight_angle_threshold

    def is_thumb_extended(self, landmarks: Sequence) -> bool:
        angle = self.calculate_angle(
            landmarks[1],
            landmarks[2],
            landmarks[4],
        )

        return angle >= self.thumb_angle_threshold

    def analyze_fingers(self, landmarks: Sequence) -> Dict[str, bool]:
        return {
            "thumb": self.is_thumb_extended(landmarks),
            "index": self.is_finger_extended(
                landmarks,
                5,
                6,
                8,
            ),
            "middle": self.is_finger_extended(
                landmarks,
                9,
                10,
                12,
            ),
            "ring": self.is_finger_extended(
                landmarks,
                13,
                14,
                16,
            ),
            "pinky": self.is_finger_extended(
                landmarks,
                17,
                18,
                20,
            ),
        }

    @staticmethod
    def is_gun_ready(fingers: Dict[str, bool]) -> bool:
        return (
            fingers["thumb"]
            and fingers["index"]
            and not fingers["middle"]
            and not fingers["ring"]
            and not fingers["pinky"]
        )

    @staticmethod
    def is_trigger_pulled(
        fingers: Dict[str, bool],
    ) -> bool:
        return (
            fingers["thumb"]
            and not fingers["index"]
            and not fingers["middle"]
            and not fingers["ring"]
            and not fingers["pinky"]
        )

    def reset(self) -> None:
        self.gun_is_ready = False
        self.trigger_is_locked = False
        self.gun_ready_at = 0.0

    def process(self, landmarks: Sequence) -> Optional[str]:
        now = time.time()

        fingers = self.analyze_fingers(landmarks)

        gun_ready_pose = self.is_gun_ready(fingers)
        trigger_pulled_pose = self.is_trigger_pulled(fingers)

        if gun_ready_pose:
            if not self.gun_is_ready:
                self.gun_ready_at = now

            self.gun_is_ready = True
            self.trigger_is_locked = False

            return "GUN READY"

        valid_trigger = (
            self.gun_is_ready
            and trigger_pulled_pose
            and not self.trigger_is_locked
            and now - self.gun_ready_at <= self.trigger_timeout
            and now - self.last_trigger_at
            >= self.trigger_cooldown
        )

        if valid_trigger:
            print("Pulled the trigger")

            pyautogui.click(button="left")

            self.last_trigger_at = now
            self.trigger_is_locked = True
            self.gun_is_ready = False

            return "PULLED THE TRIGGER"

        if (
            self.gun_is_ready
            and now - self.gun_ready_at
            > self.trigger_timeout
        ):
            self.reset()

        return None