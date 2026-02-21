from .audio import speak
from .controls import play_emotion, list_emotions, goto_target
from .vision import take_picture, save_image_person, describe_image, detect_faces, analyze_face, _IMAGES_DIR
from .movement_manager import MovementManager
from .breathing import BreathingMove
from .head_tracking_loop import start_head_tracking_loop