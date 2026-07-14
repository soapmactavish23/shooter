from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

CAMERA_INDEX = 0
SWAP_HANDEDNESS = True
WINDOW_NAME = "Gesture FPS Controller"

FACE_MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)

# Olhos: qualquer um fechado mantém o RMB pressionado.
EYE_CLOSE_THRESHOLD = 0.48
EYE_OPEN_THRESHOLD = 0.28
EYE_CLOSE_CONFIRMATION_FRAMES = 2
EYE_OPEN_CONFIRMATION_FRAMES = 2

# Reload: duas mãos abertas, uma abaixo da outra.
RELOAD_CONFIRMATION_TIME = 0.32
RELOAD_COOLDOWN = 1.25
RELOAD_MAX_HORIZONTAL_GAP = 1.25
RELOAD_MIN_VERTICAL_GAP = 0.45
RELOAD_MAX_VERTICAL_GAP = 2.60