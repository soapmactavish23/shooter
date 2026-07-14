import time
from typing import Dict, Optional, Sequence

import numpy as np

from input_controller import GameInputController


class RightHandController:
    """Reconhece a pose de pistola e envia o disparo ao jogo."""

    def __init__(
        self,
        straight_angle_threshold: float = 150.0,
        thumb_angle_threshold: float = 135.0,
        trigger_timeout: float = 4.0,
        trigger_cooldown: float = 0.20,
        click_hold_time: float = 0.08,
    ) -> None:
        self.straight_angle_threshold = straight_angle_threshold
        self.thumb_angle_threshold = thumb_angle_threshold
        self.trigger_timeout = trigger_timeout
        self.trigger_cooldown = trigger_cooldown
        self.click_hold_time = click_hold_time

        self.gun_is_ready = False
        self.trigger_is_locked = False
        self.gun_ready_at = 0.0
        self.last_trigger_at = 0.0
        self.last_fingers: Dict[str, bool] = {}

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
        return (
            self.calculate_angle(
                landmarks[mcp_index],
                landmarks[pip_index],
                landmarks[tip_index],
            )
            >= self.straight_angle_threshold
        )

    def is_thumb_extended(self, landmarks: Sequence) -> bool:
        return (
            self.calculate_angle(
                landmarks[1],
                landmarks[2],
                landmarks[4],
            )
            >= self.thumb_angle_threshold
        )

    def analyze_fingers(self, landmarks: Sequence) -> Dict[str, bool]:
        fingers = {
            "thumb": self.is_thumb_extended(landmarks),
            "index": self.is_finger_extended(landmarks, 5, 6, 8),
            "middle": self.is_finger_extended(landmarks, 9, 10, 12),
            "ring": self.is_finger_extended(landmarks, 13, 14, 16),
            "pinky": self.is_finger_extended(landmarks, 17, 18, 20),
        }
        self.last_fingers = fingers
        return fingers

    @staticmethod
    def other_fingers_folded(fingers: Dict[str, bool]) -> bool:
        return (
            not fingers["middle"]
            and not fingers["ring"]
            and not fingers["pinky"]
        )

    @classmethod
    def is_gun_ready(cls, fingers: Dict[str, bool]) -> bool:
        # O polegar ajuda a confirmar a pose, mas aceita pequenas
        # variações naturais de inclinação.
        return fingers["index"] and cls.other_fingers_folded(fingers)

    @classmethod
    def is_trigger_pulled(cls, fingers: Dict[str, bool]) -> bool:
        # O disparo é a transição do indicador aberto para fechado.
        # O polegar não é exigido, porque normalmente se move junto.
        return not fingers["index"] and cls.other_fingers_folded(fingers)

    def reset(self) -> None:
        self.gun_is_ready = False
        self.trigger_is_locked = False
        self.gun_ready_at = 0.0

    def process(self, landmarks: Sequence) -> Optional[str]:
        now = time.monotonic()
        fingers = self.analyze_fingers(landmarks)

        if self.is_gun_ready(fingers):
            if not self.gun_is_ready:
                self.gun_ready_at = now
                self.gun_is_ready = True
                self.trigger_is_locked = False
                return "ARMA PRONTA"
            return None

        valid_trigger = (
            self.gun_is_ready
            and self.is_trigger_pulled(fingers)
            and not self.trigger_is_locked
            and now - self.gun_ready_at <= self.trigger_timeout
            and now - self.last_trigger_at >= self.trigger_cooldown
        )

        if valid_trigger:
            GameInputController.left_click(self.click_hold_time)
            self.last_trigger_at = now
            self.trigger_is_locked = True
            self.gun_is_ready = False
            print("Disparo enviado ao jogo")
            return "DISPARO"

        if self.gun_is_ready and now - self.gun_ready_at > self.trigger_timeout:
            self.reset()

        return None

    def get_debug_text(self) -> str:
        if not self.last_fingers:
            return "GATILHO: aguardando mao"

        compact = "".join(
            "1" if self.last_fingers[name] else "0"
            for name in ("thumb", "index", "middle", "ring", "pinky")
        )
        return f"DEDOS P/I/M/A/MIN:{compact} | PRONTA:{self.gun_is_ready}"
