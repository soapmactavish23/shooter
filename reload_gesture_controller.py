import time
from typing import Optional, Sequence

import numpy as np

from input_controller import GameInputController


class ReloadGestureController:
    """
    Recarrega quando uma mão bate na outra, como ao encaixar o
    carregador de uma pistola.

    Acompanha a distância entre as duas mãos: quando elas se afastam
    e depois se aproximam até quase se tocar, dispara o reload uma
    única vez. É preciso afastar as mãos de novo antes que outro
    reload possa ser disparado.

    No instante exato do toque as mãos costumam se sobrepor na
    imagem da webcam, e o MediaPipe geralmente perde o rastreamento
    de uma (ou das duas) mãos por 1-2 frames bem nessa hora. Por
    isso, quando as mãos "somem" logo depois de terem sido vistas
    bem próximas, isso também conta como um toque.
    """

    def __init__(
        self,
        reload_key: str = "r",
        cooldown: float = 0.6,
        contact_ratio: float = 1.30,
        approach_ratio: float = 1.70,
        rearm_ratio: float = 2.00,
        occlusion_grace: float = 0.35,
        reload_message_duration: float = 0.5,
    ) -> None:
        self.reload_key = reload_key
        self.cooldown = cooldown
        self.contact_ratio = contact_ratio
        self.approach_ratio = approach_ratio
        self.rearm_ratio = rearm_ratio
        self.occlusion_grace = occlusion_grace
        self.reload_message_duration = reload_message_duration

        self.armed = True
        self.last_reload_at = 0.0

        self.last_gap_ratio = 0.0
        self.last_close_at = -999.0

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

    def _fire_reload(self, gap_ratio: float, reason: str) -> str:
        GameInputController.press_key(self.reload_key)
        self.last_reload_at = time.monotonic()
        self.armed = False

        print(
            f"Reload enviado ({reason}) | "
            f"gap={gap_ratio:.2f}"
        )

        return "RECARREGANDO"

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

        print(
            f"RELOAD gap={gap_ratio:.2f} "
            f"armado={self.armed}"
        )

        if gap_ratio <= self.approach_ratio:
            self.last_close_at = now

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
            return self._fire_reload(gap_ratio, "toque")

        if (
            now - self.last_reload_at
            < self.reload_message_duration
        ):
            return "RECARREGANDO"

        return None

    def handle_hands_missing(self) -> Optional[str]:
        """
        Chamado quando as duas maos nao estao sendo rastreadas ao
        mesmo tempo neste frame.

        Se elas estavam bem proximas ha pouco tempo (dentro da
        janela de tolerancia), interpreta o sumico como o proprio
        toque, ja que oclusao no momento do contato eh comum.
        """
        now = time.monotonic()

        just_saw_close = (
            now - self.last_close_at
            <= self.occlusion_grace
        )

        cooldown_finished = (
            now - self.last_reload_at >= self.cooldown
        )

        if (
            self.armed
            and cooldown_finished
            and just_saw_close
        ):
            message = self._fire_reload(
                self.last_gap_ratio,
                "oclusao no toque",
            )
            # Evita contar de novo assim que as maos reaparecerem.
            self.last_close_at = -999.0
            return message

        if (
            now - self.last_reload_at
            < self.reload_message_duration
        ):
            return "RECARREGANDO"

        # Sem as duas maos no quadro e sem sinal recente de toque,
        # considera que estao afastadas e rearma.
        if not just_saw_close:
            self.armed = True

        return None

    def reset_tracking(self) -> None:
        self.handle_hands_missing()

    def get_debug_text(self) -> str:
        armed_text = "SIM" if self.armed else "NAO"

        return (
            f"RELOAD GAP:{self.last_gap_ratio:.2f} "
            f"ARMADO:{armed_text}"
        )
