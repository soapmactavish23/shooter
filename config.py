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

# Mira: limite maior para evitar ativação ao apenas apertar os olhos.
EYE_CLOSE_THRESHOLD = 0.62
EYE_OPEN_THRESHOLD = 0.24
EYE_CLOSE_CONFIRMATION_FRAMES = 3
EYE_OPEN_CONFIRMATION_FRAMES = 3
# Frames seguidos sem blendshapes tolerados antes de soltar o botao.
# Evita que uma falha momentanea de tracking (comum durante o proprio
# piscar) solte a mira no meio do olho fechado.
EYE_MISSING_BLENDSHAPES_TOLERANCE = 6

# Recarga: duas mãos abertas e alinhadas verticalmente.
RELOAD_CONFIRMATION_TIME = 0.35
RELOAD_COOLDOWN = 1.25
RELOAD_MAX_HORIZONTAL_GAP = 1.50
RELOAD_MIN_VERTICAL_GAP = 0.25
RELOAD_MAX_VERTICAL_GAP = 3.20
