import time

import cv2
import mediapipe as mp

from eye_controller import EyeController
from left_hand_controller import LeftHandController
from right_hand_controller import RightHandController


CAMERA_INDEX = 0

# Keep this True because MediaPipe is reporting the physical
# right hand as Left and the physical left hand as Right.
SWAP_HANDEDNESS = True


def load_mediapipe_modules():
    """
    Loads MediaPipe modules while supporting installations where
    mp.solutions is not directly available.
    """
    try:
        hands_module = mp.solutions.hands
        face_mesh_module = mp.solutions.face_mesh
        drawing_utils = mp.solutions.drawing_utils
        drawing_styles = mp.solutions.drawing_styles

        return (
            hands_module,
            face_mesh_module,
            drawing_utils,
            drawing_styles,
        )

    except AttributeError:
        from mediapipe.python.solutions import drawing_styles
        from mediapipe.python.solutions import drawing_utils
        from mediapipe.python.solutions import face_mesh
        from mediapipe.python.solutions import hands

        return (
            hands,
            face_mesh,
            drawing_utils,
            drawing_styles,
        )


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """
    Opens the webcam using DirectShow on Windows.
    Falls back to the default backend when necessary.
    """
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
            "Could not open the webcam. "
            "Check whether another application is using it."
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

    return camera


def resolve_physical_hand(
    detected_hand: str,
) -> str:
    """
    Converts MediaPipe handedness into the physical hand.

    SWAP_HANDEDNESS must remain True when the mirrored camera
    causes Right and Left to be reported in reverse.
    """
    if not SWAP_HANDEDNESS:
        return detected_hand

    if detected_hand == "Right":
        return "Left"

    if detected_hand == "Left":
        return "Right"

    return detected_hand


def draw_hand(
    frame,
    hand_landmarks,
    hands_module,
    drawing_utils,
    drawing_styles,
) -> None:
    """
    Draws the hand landmarks and connections.
    """
    try:
        drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            hands_module.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )

    except AttributeError:
        drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            hands_module.HAND_CONNECTIONS,
        )


def draw_face(
    frame,
    face_landmarks,
    face_mesh_module,
    drawing_utils,
    drawing_styles,
) -> None:
    """
    Draws only the eye and face contour landmarks.
    """
    try:
        drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=face_mesh_module.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=(
                drawing_styles
                .get_default_face_mesh_contours_style()
            ),
        )

    except AttributeError:
        drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=face_mesh_module.FACEMESH_CONTOURS,
        )


def draw_interface(
    frame,
    message: str,
    detected_hands: list[str],
    face_detected: bool,
    eye_debug_text: str,
) -> None:
    """
    Draws the application status information.
    """
    height, width = frame.shape[:2]

    cv2.rectangle(
        frame,
        (0, height - 100),
        (width, height),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        message,
        (25, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    hand_text = (
        ", ".join(detected_hands)
        if detected_hands
        else "No hands"
    )

    face_text = (
        "Yes"
        if face_detected
        else "No"
    )

    cv2.putText(
        frame,
        f"Physical hands: {hand_text}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Face detected: {face_text}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Right hand: weapons 1-5 | Left hand: trigger | Blink: aim",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if eye_debug_text:
        cv2.putText(
            frame,
            eye_debug_text,
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "Press Q to exit",
        (width - 180, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    (
        hands_module,
        face_mesh_module,
        drawing_utils,
        drawing_styles,
    ) = load_mediapipe_modules()

    left_hand_controller = LeftHandController()
    right_hand_controller = RightHandController()
    eye_controller = EyeController()

    camera = open_camera(
        CAMERA_INDEX,
    )

    current_message = "SHOW YOUR HANDS AND FACE"
    message_started_at = 0.0
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
                    print(
                        "Could not capture a webcam frame."
                    )
                    break

                # Mirrored preview.
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

                face_result = face_detector.process(
                    rgb_frame,
                )

                rgb_frame.flags.writeable = True

                now = time.time()

                detected_physical_hands: list[str] = []

                left_hand_found = False
                right_hand_found = False
                face_detected = False

                eye_debug_text = ""


                # ---------------------------------------------
                # HAND DETECTION
                # ---------------------------------------------

                if (
                    hand_result.multi_hand_landmarks is not None
                    and hand_result.multi_handedness is not None
                ):
                    for hand_landmarks, handedness in zip(
                        hand_result.multi_hand_landmarks,
                        hand_result.multi_handedness,
                    ):
                        detected_hand = (
                            handedness
                            .classification[0]
                            .label
                        )

                        physical_hand = resolve_physical_hand(
                            detected_hand,
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

                        landmarks = (
                            hand_landmarks.landmark
                        )

                        # Physical right hand:
                        # selects weapons 1 to 5.
                        if physical_hand == "Right":
                            right_hand_found = True

                            action = (
                                right_hand_controller
                                .process(landmarks)
                            )

                            if action is not None:
                                current_message = action
                                message_started_at = now

                        # Physical left hand:
                        # aims and holds the trigger.
                        elif physical_hand == "Left":
                            left_hand_found = True

                            action = (
                                left_hand_controller
                                .process(landmarks)
                            )

                            if action is not None:
                                current_message = action
                                message_started_at = now


                # If the trigger hand disappears, reset it.
                # This also releases the mouse button.
                if not left_hand_found:
                    left_hand_controller.reset()

                if not right_hand_found:
                    right_hand_controller.reset()


                # ---------------------------------------------
                # FACE AND EYE DETECTION
                # ---------------------------------------------

                if face_result.multi_face_landmarks:
                    face_detected = True

                    face_landmarks_result = (
                        face_result
                        .multi_face_landmarks[0]
                    )

                    face_landmarks = (
                        face_landmarks_result.landmark
                    )

                    draw_face(
                        frame,
                        face_landmarks_result,
                        face_mesh_module,
                        drawing_utils,
                        drawing_styles,
                    )

                    eye_action = eye_controller.process(
                        face_landmarks,
                    )

                    eye_debug_text = (
                        eye_controller.get_debug_text()
                    )

                    if eye_action is not None:
                        current_message = eye_action
                        message_started_at = now

                else:
                    eye_controller.reset()
                    eye_debug_text = "Face not detected"


                # ---------------------------------------------
                # DEFAULT STATUS MESSAGE
                # ---------------------------------------------

                if (
                    now - message_started_at
                    > message_duration
                ):
                    if (
                        left_hand_found
                        and right_hand_found
                        and face_detected
                    ):
                        current_message = (
                            "WAITING FOR GESTURE"
                        )

                    elif right_hand_found:
                        current_message = (
                            "RIGHT HAND: SELECT WEAPON 1-5"
                        )

                    elif left_hand_found:
                        current_message = (
                            "LEFT HAND: HOLD THE TRIGGER"
                        )

                    elif face_detected:
                        current_message = (
                            "BLINK TO AIM"
                        )

                    else:
                        current_message = (
                            "SHOW YOUR HANDS AND FACE"
                        )


                draw_interface(
                    frame,
                    current_message,
                    detected_physical_hands,
                    face_detected,
                    eye_debug_text,
                )

                cv2.imshow(
                    "Gesture FPS Controller",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    ord("q"),
                    ord("Q"),
                ):
                    break

    except KeyboardInterrupt:
        print(
            "Execution interrupted by the user."
        )

    finally:
        # Guarantees that the mouse button is released
        # even if the program closes while firing.
        left_hand_controller.reset()

        camera.release()
        cv2.destroyAllWindows()

        for _ in range(5):
            cv2.waitKey(1)

    print(
        "Program finished."
    )


if __name__ == "__main__":
    main()
