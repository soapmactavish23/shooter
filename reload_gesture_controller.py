import time
from typing import Any

import cv2
import mediapipe as mp

from eye_controller import EyeController
from left_hand_controller import LeftHandController
from right_hand_controller import RightHandController


CAMERA_INDEX = 0
SWAP_HANDEDNESS = True
WINDOW_NAME = "Gesture FPS Controller"


def load_mediapipe_modules():
    try:
        return (
            mp.solutions.hands,
            mp.solutions.face_mesh,
            mp.solutions.drawing_utils,
            mp.solutions.drawing_styles,
        )
    except AttributeError:
        from mediapipe.python.solutions import drawing_styles
        from mediapipe.python.solutions import drawing_utils
        from mediapipe.python.solutions import face_mesh
        from mediapipe.python.solutions import hands

        return hands, face_mesh, drawing_utils, drawing_styles


def open_camera(camera_index: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError(
            "Nao foi possivel abrir a webcam. Verifique se outro programa esta usando a camera."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def resolve_physical_hand(detected_hand: str) -> str:
    if not SWAP_HANDEDNESS:
        return detected_hand
    return {"Right": "Left", "Left": "Right"}.get(detected_hand, detected_hand)


def draw_hand(frame: Any, hand_landmarks: Any, hands_module: Any, drawing_utils: Any, drawing_styles: Any) -> None:
    try:
        drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            hands_module.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )
    except AttributeError:
        drawing_utils.draw_landmarks(frame, hand_landmarks, hands_module.HAND_CONNECTIONS)


def draw_face(frame: Any, face_landmarks: Any, face_mesh_module: Any, drawing_utils: Any, drawing_styles: Any) -> None:
    try:
        drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=face_mesh_module.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
    except AttributeError:
        drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=face_mesh_module.FACEMESH_CONTOURS,
        )


def draw_interface(
    frame: Any,
    message: str,
    detected_hands: list[str],
    eye_debug: str,
    face_found: bool,
) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - 105), (width, height), (0, 0, 0), -1)

    hand_text = ", ".join(sorted(set(detected_hands))) if detected_hands else "nenhuma"
    face_text = "detectado" if face_found else "nao detectado"

    cv2.putText(frame, f"Maos: {hand_text} | Rosto: {face_text}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Esquerda: armas 1-5 | Direita: pistola/gatilho | Piscar: mira", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, eye_debug, (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, message, (25, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Q: sair | R: recalibrar olhos", (width - 320, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    hands_module, face_mesh_module, drawing_utils, drawing_styles = load_mediapipe_modules()

    left_hand_controller = LeftHandController()
    right_hand_controller = RightHandController()
    # False = clique direito por piscada.
    # True = primeira piscada segura RMB; segunda piscada libera.
    eye_controller = EyeController(aim_hold_mode=False)
    camera = open_camera(CAMERA_INDEX)

    current_message = "MANTENHA OS OLHOS ABERTOS PARA CALIBRAR"
    message_started_at = time.monotonic()
    message_duration = 1.30

    try:
        with hands_module.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        ) as hand_detector, face_mesh_module.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        ) as face_detector:
            while True:
                success, frame = camera.read()
                if not success or frame is None:
                    print("Nao foi possivel capturar um frame da webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                hand_result = hand_detector.process(rgb_frame)
                face_result = face_detector.process(rgb_frame)
                rgb_frame.flags.writeable = True

                now = time.monotonic()
                detected_physical_hands: list[str] = []
                left_hand_found = False
                right_hand_found = False
                face_found = False

                if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                    for hand_landmarks, handedness in zip(
                        hand_result.multi_hand_landmarks,
                        hand_result.multi_handedness,
                    ):
                        detected_hand = handedness.classification[0].label
                        physical_hand = resolve_physical_hand(detected_hand)
                        detected_physical_hands.append(physical_hand)

                        draw_hand(
                            frame,
                            hand_landmarks,
                            hands_module,
                            drawing_utils,
                            drawing_styles,
                        )

                        landmarks = hand_landmarks.landmark
                        action = None

                        if physical_hand == "Left":
                            left_hand_found = True
                            action = left_hand_controller.process(landmarks)
                        elif physical_hand == "Right":
                            right_hand_found = True
                            action = right_hand_controller.process(landmarks)

                        if action:
                            current_message = action
                            message_started_at = now

                if not left_hand_found:
                    left_hand_controller.reset()
                if not right_hand_found:
                    right_hand_controller.reset()

                if face_result.multi_face_landmarks:
                    face_found = True
                    face_landmarks = face_result.multi_face_landmarks[0]
                    draw_face(
                        frame,
                        face_landmarks,
                        face_mesh_module,
                        drawing_utils,
                        drawing_styles,
                    )
                    eye_action = eye_controller.process(face_landmarks.landmark)
                    if eye_action:
                        current_message = eye_action
                        message_started_at = now
                else:
                    eye_controller.reset()

                if now - message_started_at > message_duration:
                    if not eye_controller.is_calibrated:
                        current_message = "MANTENHA OS OLHOS ABERTOS PARA CALIBRAR"
                    elif left_hand_found and right_hand_found:
                        current_message = "AGUARDANDO GESTO"
                    elif right_hand_found:
                        current_message = "MAO DIREITA: FORME A PISTOLA E PUXE O GATILHO"
                    elif left_hand_found:
                        current_message = "MAO ESQUERDA: MOSTRE UM NUMERO DE 1 A 5"
                    else:
                        current_message = "MOSTRE AS MAOS"

                draw_interface(
                    frame,
                    current_message,
                    detected_physical_hands,
                    eye_controller.get_debug_text(),
                    face_found,
                )
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("r"), ord("R")):
                    eye_controller.reset_calibration()
                    current_message = "RECALIBRANDO OLHOS"
                    message_started_at = now

    except KeyboardInterrupt:
        print("Execucao interrompida pelo usuario.")
    finally:
        eye_controller.release_buttons()
        camera.release()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

    print("Programa finalizado.")


if __name__ == "__main__":
    main()