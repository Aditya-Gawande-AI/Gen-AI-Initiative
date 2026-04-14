import io
from PIL import Image
import pytesseract

def ocr_image_bytes(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    # You can optionally do img = img.convert("L") for better OCR sometimes
    text = pytesseract.image_to_string(img)
    return text or ""