import base64
import io
import re
from PIL import Image

from langchain_core.messages import HumanMessage
from gen_ai_hub.proxy.langchain.init_models import init_llm  # SAP GenAI Hub wrapper


def _pick_serial(text: str) -> str:
    """Pick the most likely serial token from model output."""
    if not text:
        return ""

    t = text.strip().upper()

    # Prefer explicit labels like S/N, SN, SERIAL, SERIAL NO
    labeled = re.findall(r"(?:S\/N|SN|SERIAL(?:\s*NO)?)\s*[:#\-]?\s*([A-Z0-9\-]{5,})", t)
    if labeled:
        return labeled[0].strip()

    # Fallback: any long alphanumeric token
    tokens = re.findall(r"[A-Z0-9\-]{8,}", t)
    return tokens[0].strip() if tokens else text.strip()


def _compress_for_vision(image_bytes: bytes, max_side: int = 768, jpeg_quality: int = 60) -> bytes:
    """
    Downscale + convert to JPEG to reduce payload.
    Keeps request small and avoids huge context usage.
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB (JPEG needs RGB)
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
    return out.getvalue()


def extract_serial_from_image_bytes(image_bytes: bytes, filename: str = "image.jpg") -> str:
    """
    Agent 2: Send IMAGE to LLM (vision input) and extract Serial Number.
    This matches the task requirement: read image -> send to LLM -> capture serial. [2](https://outlook.office365.com/owa/?ItemID=AAMkADYxZjRlYzJjLWVlMjgtNDFjOS05NDQzLTk2MzRjODgwYzEwNABGAAAAAAA0lwKjoj8GToBLllHOlVrRBwBV0tJW8KfPTpGHPB2sLYqOAAAAAAEMAABV0tJW8KfPTpGHPB2sLYqOAAA33IY%2bAAA%3d&exvsurl=1&viewmodel=ReadMessageItem)
    """
    llm = init_llm("gpt-4o", max_tokens=60)

    # Compress to reduce payload
    small_jpg = _compress_for_vision(image_bytes)

    # Encode for data URL
    b64 = base64.b64encode(small_jpg).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    prompt_text = (
        "Extract the PRODUCT SERIAL NUMBER from this image.\n"
        "Rules:\n"
        "- Return ONLY the serial number (no extra words).\n"
        "- If multiple candidates exist, choose the one labeled Serial / Serial No / S/N / SN.\n"
        "- If you cannot find any serial number, return exactly: NOT_FOUND\n"
        f"(Filename: {filename})"
    )

    # IMPORTANT: Send image as vision content, NOT as plain text base64
    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    resp = llm.invoke([msg])
    raw = (resp.content or "").strip()

    serial = _pick_serial(raw)
    if "NOT_FOUND" in serial.upper():
        return ""

    return serial