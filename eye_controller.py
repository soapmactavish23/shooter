from typing import Iterable, Optional

from input_controller import GameInputController


class EyeController:
    """
    Controla a mira pelos blendshapes do Face Landmarker.

    eyeBlinkLeft ou eyeBlinkRight acima do limite:
        mantém o botão direito pressionado.

    Os dois olhos abaixo do limite de abertura:
        libera o botão direito.
    """

    LEFT_BLINK_NAMES = {
        "eyeBlinkLeft",
        "eye_blink_left",
    }
    RIGHT_BLINK_NAMES = {
        "eyeBlinkRight",
        "eye_blink_right",
    }

    def __init__(
        self,
        close_threshold: float = 0.48,
        open_threshold: float = 0.28,
        close_confirmation_frames: int = 2,
        open_confirmation_frames: int = 2,
        missing_blendshapes_tolerance: int = 6,
    ) -> None:
        if open_threshold >= close_threshold:
            raise ValueError(
                "open_threshold deve ser menor que close_threshold."
            )

        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.close_confirmation_frames = close_confirmation_frames
        self.open_confirmation_frames = open_confirmation_frames

        # Quantos frames seguidos sem blendshapes toleramos antes de
        # soltar o botao. Um piscar de olhos costuma derrubar a deteccao
        # de rosto por 1-2 frames, e sem essa tolerancia a mira soltava
        # e reengatava a cada piscada, dando a sensacao de "nao segurar".
        self.missing_blendshapes_tolerance = (
            missing_blendshapes_tolerance
        )
        self.missing_blendshapes_count = 0

        self.left_blink_score = 0.0
        self.right_blink_score = 0.0

        self.closed_frame_count = 0
        self.open_frame_count = 0
        self.aim_is_active = False

    @staticmethod
    def category_name(category) -> str:
        return str(
            getattr(
                category,
                "category_name",
                getattr(
                    category,
                    "display_name",
                    "",
                ),
            )
        )

    @staticmethod
    def category_score(category) -> float:
        return float(
            getattr(
                category,
                "score",
                0.0,
            )
        )

    def extract_blink_scores(
        self,
        categories: Iterable,
    ) -> tuple[float, float]:
        left_score = 0.0
        right_score = 0.0

        for category in categories:
            name = self.category_name(category)
            score = self.category_score(category)

            if name in self.LEFT_BLINK_NAMES:
                left_score = score
            elif name in self.RIGHT_BLINK_NAMES:
                right_score = score

        self.left_blink_score = left_score
        self.right_blink_score = right_score

        return left_score, right_score

    def press_aim(self) -> Optional[str]:
        if self.aim_is_active:
            return None

        GameInputController.right_button_down()
        self.aim_is_active = True

        print(
            "Mira ativada | "
            f"eyeBlinkLeft={self.left_blink_score:.2f} | "
            f"eyeBlinkRight={self.right_blink_score:.2f}"
        )

        return "MIRA ATIVADA"

    def release_aim(self) -> Optional[str]:
        if not self.aim_is_active:
            return None

        GameInputController.right_button_up()
        self.aim_is_active = False

        print("Mira desativada")

        return "MIRA DESATIVADA"

    def handle_missing_blendshapes(self) -> Optional[str]:
        """
        Chamado quando o frame atual nao trouxe blendshapes de rosto.

        Tolera algumas falhas seguidas (comuns durante o proprio
        piscar/fechar do olho) antes de soltar o botao, em vez de
        resetar o estado a cada frame perdido.
        """
        self.missing_blendshapes_count += 1

        if (
            self.missing_blendshapes_count
            < self.missing_blendshapes_tolerance
        ):
            return None

        self.closed_frame_count = 0
        self.open_frame_count = 0

        return self.release_aim()

    def process(
        self,
        categories: Iterable,
    ) -> Optional[str]:
        self.missing_blendshapes_count = 0

        left_score, right_score = (
            self.extract_blink_scores(
                categories,
            )
        )

        any_eye_closed = (
            left_score >= self.close_threshold
            or right_score >= self.close_threshold
        )

        both_eyes_open = (
            left_score <= self.open_threshold
            and right_score <= self.open_threshold
        )

        if any_eye_closed:
            self.closed_frame_count += 1
            self.open_frame_count = 0

            if (
                self.closed_frame_count
                >= self.close_confirmation_frames
            ):
                return self.press_aim()

            return None

        if both_eyes_open:
            self.open_frame_count += 1
            self.closed_frame_count = 0

            if (
                self.open_frame_count
                >= self.open_confirmation_frames
            ):
                return self.release_aim()

            return None

        # Histerese: mantém o estado atual quando os valores ficam
        # entre o limite de abertura e o limite de fechamento.
        self.closed_frame_count = 0
        self.open_frame_count = 0
        return None

    def reset(self) -> None:
        self.closed_frame_count = 0
        self.open_frame_count = 0
        self.left_blink_score = 0.0
        self.right_blink_score = 0.0
        self.release_aim()

    def release_buttons(self) -> None:
        self.release_aim()

    def get_debug_text(self) -> str:
        aim_state = (
            "ON"
            if self.aim_is_active
            else "OFF"
        )

        return (
            f"BLINK E:{self.left_blink_score:.2f} "
            f"D:{self.right_blink_score:.2f} | "
            f"MIRA:{aim_state}"
        )