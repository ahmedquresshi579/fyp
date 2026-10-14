import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from spellchecker import SpellChecker
from PIL import Image
import cv2

print("Loading handwriting AI... this takes a moment")
_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
_spell = SpellChecker()
print("Ready")


def recognize_text(image_bgr):
    """Takes an OpenCV (BGR) image, returns (raw_text, corrected_text)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    pixel_values = _processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = _model.generate(
        pixel_values,
        max_new_tokens=30,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    raw_text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    words = raw_text.split()
    corrected_words = []
    for w in words:
        if w.lower() in _spell:
            corrected_words.append(w)
        else:
            fix = _spell.correction(w)
            corrected_words.append(fix if fix else w)

    corrected_text = " ".join(corrected_words)
    corrected_text = corrected_text.strip(" .,;:!?")
    return raw_text, corrected_text