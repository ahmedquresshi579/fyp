import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

files = sorted(
    f for f in os.listdir("samples")
    if f.endswith(".png")
)

for filename in files:
    image = Image.open(
        f"samples/{filename}"
    ).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=50
    )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    print(f"{filename}: {text}")