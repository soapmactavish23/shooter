import time

try:
    import pydirectinput

    pydirectinput.PAUSE = 0
    pydirectinput.FAILSAFE = False
    DIRECT_INPUT_AVAILABLE = True
except ImportError:
    pydirectinput = None
    DIRECT_INPUT_AVAILABLE = False

import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


class GameInputController:
    """Envia teclado e mouse para jogos usando DirectInput quando disponível."""

    @staticmethod
    def press_key(key: str) -> None:
        if DIRECT_INPUT_AVAILABLE:
            pydirectinput.press(key)
        else:
            pyautogui.press(key)

    @staticmethod
    def left_click(hold_time: float = 0.04) -> None:
        if DIRECT_INPUT_AVAILABLE:
            pydirectinput.mouseDown(button="left")
            time.sleep(hold_time)
            pydirectinput.mouseUp(button="left")
        else:
            pyautogui.mouseDown(button="left")
            time.sleep(hold_time)
            pyautogui.mouseUp(button="left")

    @staticmethod
    def right_click(hold_time: float = 0.08) -> None:
        if DIRECT_INPUT_AVAILABLE:
            pydirectinput.mouseDown(button="right")
            time.sleep(hold_time)
            pydirectinput.mouseUp(button="right")
        else:
            pyautogui.mouseDown(button="right")
            time.sleep(hold_time)
            pyautogui.mouseUp(button="right")

    @staticmethod
    def right_button_down() -> None:
        if DIRECT_INPUT_AVAILABLE:
            pydirectinput.mouseDown(button="right")
        else:
            pyautogui.mouseDown(button="right")

    @staticmethod
    def right_button_up() -> None:
        if DIRECT_INPUT_AVAILABLE:
            pydirectinput.mouseUp(button="right")
        else:
            pyautogui.mouseUp(button="right")