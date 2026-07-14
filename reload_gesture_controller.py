import time
from typing import Optional, Sequence

import numpy as np

from input_controller import GameInputController


class ReloadGestureController:
    """
    Recarrega quando uma mão bate na outra, como ao encaixar o
    carregador de uma pistola.

    Em vez de exigir uma pose parada, acompanha a distância entre as
    duas mãos: quando elas se afastam e depois se aproximam até
    quase se tocar, dispara o reload uma única vez. É preciso afastar
    as mãos de novo antes que outro reload possa ser disparado, o que
    evita disparos repetidos enquanto as mãos ficam coladas.
    """

    def __init__(
        self,
        reload_key: str = "r",
        cooldown: float = 0.6,
        contact_ratio: float = 1.10,
        rearm_ratio: float = 2.00,
        reload_message_duration: float = 0.5,
    ) -> None:
        self.reload_key = reload_key
        self.cooldown = cooldown
        self.contact_ratio = contact_ratio
        self.rearm_ratio = rearm_ratio
        self.reload_message_duration = reload_message_duration

        self.armed = True
        self.last_reload_at = 0.0

        self.last_gap_ratio = 0.0

    @staticmethod
    def point(landmark) -> np.ndarray:
        return np.array([landmark.x, landmark.y, landmark.z], dtype=float)

    def palm_center(self, landmarks: Sequence) -> np.ndarray:
        return np.mean(
            [self.point(landmarks[index])[:2] for index in (0, 5, 9, 13, 17)],
            axis=0,
        )

    def palm_width(self, landmarks: Sequence) -> float:
        return float(
            np.linalg.norm(
                self.point(landmarks[5])[:2] - self.point(landmarks[17])[:2]
            )
        )

    def process(
        self,
        left_landmarks: Sequence,
        right_landmarks: Sequence,
    ) -> Optional[str]:
        now = time.monotonic()

        left_center = self.palm_center(left_landmarks)
        right_center = self.palm_center(right_landmarks)

        gap = float(np.linalg.norm(left_center - right_center))

        average_width = max(
            (
                self.palm_width(left_landmarks)
                + self.palm_width(right_landmarks)
            )
            / 2.0,
            0.001,
        )

        gap_ratio = gap / average_width
        self.last_gap_ratio = gap_ratio

        # As maos precisam se afastar o suficiente para "armar" o
        # proximo toque. Sem isso, encostar e ficar parado dispararia
        # reload sem fim.
        if not self.armed and gap_ratio >= self.rearm_ratio:
            self.armed = True

        cooldown_finished = (
            now - self.last_reload_at >= self.cooldown
        )

        if (
            self.armed
            and cooldown_finished
            and gap_ratio <= self.contact_ratio
        ):
            GameInputController.press_key(self.reload_key)
            self.last_reload_at = now
            self.armed = False

            print(
                "Reload enviado (mao bateu na mao) | "
                f"gap={gap_ratio:.2f}"
            )

            return "RECARREGANDO"

        if (
            now - self.last_reload_at
            < self.reload_message_duration
        ):
            return "RECARREGANDO"

        return None

    def reset_tracking(self) -> None:
        # Sem as duas maos no quadro nao ha como medir distancia.
        # Rearma por seguranca, ja que perder o rastreamento costuma
        # significar que as maos nao estao mais coladas.
        self.armed = True

    def get_debug_text(self) -> str:
        armed_text = "SIM" if self.armed else "NAO"

        return (
            f"RELOAD GAP:{self.last_gap_ratio:.2f} "
            f"ARMADO:{armed_text}"
        )
