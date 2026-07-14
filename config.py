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

# Mira: reage a qualquer olho fechado (piscada/wink), sem exigir
# frames seguidos de confirmação. Se a mira não estiver ativando,
# observe os valores "BLINK E/D" na tela: eles mostram o quanto o
# Face Landmarker está lendo cada olho fechado, e este limite deve
# ficar um pouco abaixo do valor que você observar com o olho fechado.
EYE_CLOSE_THRESHOLD = 0.40
EYE_OPEN_THRESHOLD = 0.24
# Frames seguidos sem blendshapes tolerados antes de soltar o botao.
# Evita que uma falha momentanea de tracking (comum durante o proprio
# piscar) solte a mira no meio do olho fechado.
EYE_MISSING_BLENDSHAPES_TOLERANCE = 6

# Recarga: uma mão bate na outra (gesto de encaixar o carregador).
# RELOAD_CONTACT_RATIO: distancia (em larguras de palma) considerada
# "toque". Menor = precisa encostar mais.
# RELOAD_REARM_RATIO: distancia minima para as maos serem consideradas
# "afastadas" de novo e destravar o proximo reload.
# Se não estiver disparando, observe "RELOAD GAP" na tela: é a
# distância real medida quando as mãos se tocam. Ajuste
# RELOAD_CONTACT_RATIO para um pouco acima desse valor.
RELOAD_COOLDOWN = 0.6
RELOAD_CONTACT_RATIO = 1.10
RELOAD_REARM_RATIO = 2.002