import time
from typing import Any

import cv2
import mediapipe as mp

from config import (
    CAMERA_INDEX,
    EYE_CLOSE_THRESHOLD,
    EYE_MISSING_BLENDSHAPES_TOLERANCE,
    EYE_OPEN_THRESHOLD,
    FACE_MODEL_PATH,
    FACE_MODEL_URL,
    RELOAD_CONFIRMATION_TIME,
    RELOAD_COOLDOWN,
    RELOAD_MAX_HORIZONTAL_GAP,
    RELOAD_MAX_VERTICAL_GAP,
    RELOAD_MIN_VERTICAL_GAP,
    SWAP_HANDEDNESS,
    WINDOW_NAME,
)
from eye_controller import EyeController
from left_hand_controller import LeftHandController
from model_manager import ensure_model
from reload_gesture_controller import ReloadGestureController
from right_hand_controller import RightHandController


def load_hand_modules():
    try:
        return (
            mp.solutions.hands,
            mp.solutions.drawing_utils,
            mp.solutions.drawing_styles,
        )
    except AttributeError:
        from mediapipe.python.solutions import drawing_styles
        from mediapipe.python.solutions import drawing_utils
        from mediapipe.python.solutions import hands

        return (
            hands,
            drawing_utils,
            drawing_styles,
        )


def create_face_landmarker():
    model_path = ensure_model(
        FACE_MODEL_PATH,
        FACE_MODEL_URL,
    )

    # Em algumas versões do MediaPipe no Windows, model_asset_path
    # concatena incorretamente o caminho absoluto com site-packages.
    # Carregar o modelo como bytes evita esse problema.
    base_options = mp.tasks.BaseOptions(
        model_asset_buffer=model_path.read_bytes(),
    )

    options = (
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                mp.tasks.vision.RunningMode.VIDEO
            ),
            num_faces=1,
            min_face_detection_confidence=0.50,
            min_face_presence_confidence=0.50,
            min_tracking_confidence=0.50,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
    )

    return (
        mp.tasks.vision.FaceLandmarker
        .create_from_options(options)
    )


def open_camera(
    camera_index: int,
) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(
        camera_index,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(
            camera_index,
        )

    if not camera.isOpened():
        raise RuntimeError(
            "Nao foi possivel abrir a webcam."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )
    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )
    camera.set(
        cv2.CAP_PROP_FPS,
        30,
    )
    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    return camera


def resolve_physical_hand(
    detected_hand: str,
) -> str:
    if not SWAP_HANDEDNESS:
        return detected_hand

    return {
        "Right": "Left",
        "Left": "Right",
    }.get(
        detected_hand,
        detected_hand,
    )


def draw_hand(
    frame: Any,
    hand_landmarks: Any,
    hands_module: Any,
    drawing_utils: Any,
    drawing_styles: Any,
) -> None:
    try:
        drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            hands_module.HAND_CONNECTIONS,
            drawing_styles
            .get_default_hand_landmarks_style(),
            drawing_styles
            .get_default_hand_connections_style(),
        )
    except AttributeError:
        drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            hands_module.HAND_CONNECTIONS,
        )


def draw_face_landmarks(
    frame: Any,
    face_landmarks: list,
) -> None:
    height, width = frame.shape[:2]

    # Desenha apenas os pontos principais dos olhos para reduzir custo.
    eye_indexes = (
        33,
        133,
        159,
        145,
        362,
        263,
        386,
        374,
    )

    for index in eye_indexes:
        if index >= len(face_landmarks):
            continue

        landmark = face_landmarks[index]
        x = int(
            landmark.x * width
        )
        y = int(
            landmark.y * height
        )

        cv2.circle(
            frame,
            (x, y),
            2,
            (255, 255, 255),
            -1,
        )


def draw_interface(
    frame: Any,
    message: str,
    detected_hands: list[str],
    eye_debug: str,
    reload_debug: str,
    face_found: bool,
) -> None:
    height, width = frame.shape[:2]

    cv2.rectangle(
        frame,
        (0, height - 132),
        (width, height),
        (0, 0, 0),
        -1,
    )

    hands_text = (
        ", ".join(
            sorted(
                set(detected_hands)
            )
        )
        if detected_hands
        else "nenhuma"
    )

    cv2.putText(
        frame,
        (
            f"Maos: {hands_text} | "
            f"Rosto: "
            f"{'detectado' if face_found else 'nao detectado'}"
        ),
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        (
            "Armas: esquerda | "
            "Disparo: direita | "
            "Mira: qualquer olho fechado | "
            "Reload: duas maos abertas empilhadas"
        ),
        (18, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        eye_debug,
        (18, 87),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        reload_debug,
        (18, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        message,
        (24, height - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Q: sair",
        (width - 105, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    (
        hands_module,
        drawing_utils,
        drawing_styles,
    ) = load_hand_modules()

    eye_controller = EyeController(
        close_threshold=EYE_CLOSE_THRESHOLD,
        open_threshold=EYE_OPEN_THRESHOLD,
        missing_blendshapes_tolerance=(
            EYE_MISSING_BLENDSHAPES_TOLERANCE
        ),
    )

    reload_controller = ReloadGestureController(
        confirmation_time=(
            RELOAD_CONFIRMATION_TIME
        ),
        cooldown=RELOAD_COOLDOWN,
        max_horizontal_gap_ratio=(
            RELOAD_MAX_HORIZONTAL_GAP
        ),
        min_vertical_gap_ratio=(
            RELOAD_MIN_VERTICAL_GAP
        ),
        max_vertical_gap_ratio=(
            RELOAD_MAX_VERTICAL_GAP
        ),
    )

    left_hand_controller = (
        LeftHandController()
    )
    right_hand_controller = (
        RightHandController()
    )

    camera = open_camera(
        CAMERA_INDEX,
    )

    current_message = (
        "AGUARDANDO GESTOS"
    )
    message_started_at = (
        time.monotonic()
    )
    message_duration = 1.20

    video_started_at = time.monotonic()
    last_frame_timestamp_ms = -1

    try:
        with hands_module.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60,
        ) as hand_detector, create_face_landmarker() as face_landmarker:
            while True:
                success, frame = camera.read()

                if (
                    not success
                    or frame is None
                ):
                    print(
                        "Falha ao capturar frame."
                    )
                    break

                frame = cv2.flip(
                    frame,
                    1,
                )

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )
                rgb_frame.flags.writeable = False

                hand_result = hand_detector.process(
                    rgb_frame,
                )

                frame_timestamp_ms = int(
                    (time.monotonic() - video_started_at) * 1000
                )
                if frame_timestamp_ms <= last_frame_timestamp_ms:
                    frame_timestamp_ms = last_frame_timestamp_ms + 1
                last_frame_timestamp_ms = frame_timestamp_ms

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                face_result = (
                    face_landmarker
                    .detect_for_video(
                        mp_image,
                        frame_timestamp_ms,
                    )
                )

                rgb_frame.flags.writeable = True

                now = time.monotonic()

                detected_physical_hands: list[str] = []
                left_hand_landmarks = None
                right_hand_landmarks = None

                if (
                    hand_result.multi_hand_landmarks
                    and hand_result.multi_handedness
                ):
                    for (
                        hand_landmarks,
                        handedness,
                    ) in zip(
                        hand_result.multi_hand_landmarks,
                        hand_result.multi_handedness,
                    ):
                        detected_hand = (
                            handedness
                            .classification[0]
                            .label
                        )

                        physical_hand = (
                            resolve_physical_hand(
                                detected_hand,
                            )
                        )

                        detected_physical_hands.append(
                            physical_hand,
                        )

                        draw_hand(
                            frame,
                            hand_landmarks,
                            hands_module,
                            drawing_utils,
                            drawing_styles,
                        )

                        if physical_hand == "Left":
                            left_hand_landmarks = (
                                hand_landmarks.landmark
                            )
                        elif physical_hand == "Right":
                            right_hand_landmarks = (
                                hand_landmarks.landmark
                            )

                left_hand_found = (
                    left_hand_landmarks
                    is not None
                )
                right_hand_found = (
                    right_hand_landmarks
                    is not None
                )

                # RELOAD TEM PRIORIDADE.
                # Quando as duas mãos abertas ficam empilhadas,
                # não executa troca de arma nem disparo.
                reload_action = None

                if (
                    left_hand_found
                    and right_hand_found
                ):
                    reload_action = (
                        reload_controller.process(
                            left_hand_landmarks,
                            right_hand_landmarks,
                        )
                    )
                else:
                    reload_controller.reset_tracking()

                reload_pose_active = (
                    reload_action is not None
                )

                if reload_pose_active:
                    left_hand_controller.reset()
                    right_hand_controller.reset()

                    current_message = reload_action
                    message_started_at = now
                else:
                    if left_hand_found:
                        left_action = (
                            left_hand_controller.process(
                                left_hand_landmarks,
                            )
                        )

                        if left_action:
                            current_message = left_action
                            message_started_at = now
                    else:
                        left_hand_controller.reset()

                    if right_hand_found:
                        right_action = (
                            right_hand_controller.process(
                                right_hand_landmarks,
                            )
                        )

                        if right_action:
                            current_message = right_action
                            message_started_at = now
                    else:
                        right_hand_controller.reset()

                face_found = bool(
                    face_result.face_landmarks
                )

                if face_found:
                    draw_face_landmarks(
                        frame,
                        face_result.face_landmarks[0],
                    )

                blendshapes_found = bool(
                    face_result.face_blendshapes
                )

                if blendshapes_found:
                    eye_action = (
                        eye_controller.process(
                            face_result
                            .face_blendshapes[0]
                        )
                    )

                    if eye_action:
                        current_message = eye_action
                        message_started_at = now
                else:
                    eye_action = (
                        eye_controller
                        .handle_missing_blendshapes()
                    )

                    if eye_action:
                        current_message = eye_action
                        message_started_at = now

                if (
                    now - message_started_at
                    > message_duration
                ):
                    if reload_pose_active:
                        current_message = (
                            "MANTENHA AS MAOS EMPILHADAS"
                        )
                    elif (
                        left_hand_found
                        and right_hand_found
                    ):
                        current_message = (
                            "AGUARDANDO GESTO"
                        )
                    elif right_hand_found:
                        current_message = (
                            "MAO DIREITA: DISPARO"
                        )
                    elif left_hand_found:
                        current_message = (
                            "MAO ESQUERDA: ARMAS 1-5"
                        )
                    else:
                        current_message = (
                            "MOSTRE AS MAOS"
                        )

                draw_interface(
                    frame,
                    current_message,
                    detected_physical_hands,
                    eye_controller
                    .get_debug_text(),
                    reload_controller
                    .get_debug_text(),
                    face_found,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    frame,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    ord("Q"),
                ):
                    break

    except KeyboardInterrupt:
        print(
            "Execucao interrompida."
        )
    finally:
        eye_controller.release_buttons()
        camera.release()
        cv2.destroyAllWindows()

        for _ in range(5):
            cv2.waitKey(1)

    print("Programa finalizado.")


if __name__ == "__main__":
    main()