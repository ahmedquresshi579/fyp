from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/times.ttf",   # Windows Times New Roman
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",  # Linux fallback
]

def _load_font(size):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_clean_text(text, width, height, font_size=48):
    """Returns a white-background BGR OpenCV image with the given text
    drawn in a clean serif font. Supports multiple lines separated by \\n."""
    pil_image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(pil_image)
    font = _load_font(font_size)

    y = 30
    line_height = int(font_size * 1.4)
    for line in text.split("\n"):
        if line:
            draw.text((30, y), line, fill=(20, 20, 20), font=font)
        y += line_height

    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return bgr_image