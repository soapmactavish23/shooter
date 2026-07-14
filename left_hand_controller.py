import time
from typing import Dict, Optional, Sequence

import numpy as np
from input_controller import GameInputController


class LeftHandController:
    """Seleciona as armas 1 a 5 com a mão esquerda."""

    def __init__(
        self,
        straight_angle_threshold: float = 155.0,
        thumb_angle_threshold: float = 145.0,
        confirmation_time: float = 0.40,
    ) -> None:
        self.straight_angle_threshold = straight_angle_threshold
        self.thumb_angle_threshold = thumb_angle_threshold
        self.confirmation_time = confirmation_time

        self.number_candidate: Optional[int] = None
        self.confirmed_number: Optional[int] = None
        self.candidate_started_at = 0.0

    @staticmethod
    def calculate_angle(point_a, point_b, point_c) -> float:
        vector_ba = np.array(
            [point_a.x - point_b.x, point_a.y - point_b.y, point_a.z - point_b.z],
            dtype=float,
        )
        vector_bc = np.array(
            [point_c.x - point_b.x, point_c.y - point_b.y, point_c.z - point_b.z],
            dtype=float,
        )

        denominator = np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)
        if denominator == 0:
            return 0.0

        cosine = np.clip(np.dot(vector_ba, vector_bc) / denominator, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def is_finger_extended(
        self,
        landmarks: Sequence,
        mcp_index: int,
        pip_index: int,
        tip_index: int,
    ) -> bool:
        angle = self.calculate_angle(
            landmarks[mcp_index], landmarks[pip_index], landmarks[tip_index]
        )
        return angle >= self.straight_angle_threshold

    def is_thumb_extended(self, landmarks: Sequence) -> bool:
        angle = self.calculate_angle(landmarks[1], landmarks[2], landmarks[4])
        return angle >= self.thumb_angle_threshold

    def analyze_fingers(self, landmarks: Sequence) -> Dict[str, bool]:
        return {
            "thumb": self.is_thumb_extended(landmarks),
            "index": self.is_finger_extended(landmarks, 5, 6, 8),
            "middle": self.is_finger_extended(landmarks, 9, 10, 12),
            "ring": self.is_finger_extended(landmarks, 13, 14, 16),
            "pinky": self.is_finger_extended(landmarks, 17, 18, 20),
        }

    @staticmethod
    def recognize_number(fingers: Dict[str, bool]) -> Optional[int]:
        thumb = fingers["thumb"]
        index = fingers["index"]
        middle = fingers["middle"]
        ring = fingers["ring"]
        pinky = fingers["pinky"]

        if index and not thumb and not middle and not ring and not pinky:
            return 1
        if index and middle and not thumb and not ring and not pinky:
            return 2
        if index and middle and ring and not thumb and not pinky:
            return 3
        if not thumb and index and middle and ring and pinky:
            return 4
        if thumb and index and middle and ring and pinky:
            return 5
        return None

    def reset(self) -> None:
        self.number_candidate = None
        self.confirmed_number = None
        self.candidate_started_at = 0.0

    def process(self, landmarks: Sequence) -> Optional[str]:
        now = time.monotonic()
        current_number = self.recognize_number(self.analyze_fingers(landmarks))

        if current_number is None:
            self.reset()
            return None

        if current_number != self.number_candidate:
            self.number_candidate = current_number
            self.confirmed_number = None
            self.candidate_started_at = now
            return None

        stable = now - self.candidate_started_at >= self.confirmation_time
        if stable and current_number != self.confirmed_number:
            GameInputController.press_key(str(current_number))
            self.confirmed_number = current_number
            print(f"Weapon {current_number} selected")
            return f"ARMA {current_number} SELECIONADA"

        return None