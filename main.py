import os
import io
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageEnhance
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
app = FastAPI(title="Office Presence Shop API")

# ─── Paths ────────────────────────────────────────────────────────────────────
# catalog/ folder must sit next to main.py on the backend server
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(BASE_DIR, "catalog")

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Set ALLOWED_ORIGINS in your server env, comma-separated
# e.g. ALLOWED_ORIGINS=https://your-frontend.vercel.app,https://your-custom-domain.com
# Defaults to localhost:5173 (Vite dev) and localhost:3000 (CRA dev)
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─── Gemini ───────────────────────────────────────────────────────────────────
gemini_client = genai.Client()

MAX_RETRIES   = 3
TARGET_SIZE   = (768, 1024)
CLOTHING_SIZE = (600, 600)

# ─── Allowed filenames (whitelist against path traversal) ─────────────────────
def safe_catalog_path(subfolder: str, filename: str) -> str:
    """
    Resolves the catalog path and raises 400 if the filename tries to
    escape the catalog directory (e.g. ../../etc/passwd).
    """
    # Strip any directory components the client might inject
    safe_name = os.path.basename(filename)
    full_path  = os.path.realpath(os.path.join(CATALOG_PATH, subfolder, safe_name))
    allowed    = os.path.realpath(os.path.join(CATALOG_PATH, subfolder))

    if not full_path.startswith(allowed + os.sep):
        raise HTTPException(status_code=400, detail="Invalid file path.")
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail=f"Catalog item not found: {safe_name}")
    return full_path


# ─── Image helpers ────────────────────────────────────────────────────────────
def normalize_image(
    img: Image.Image,
    target_size: tuple,
    keep_aspect: bool = True,
    boost_saturation: bool = False,
) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode == "P":
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if boost_saturation:
        img = ImageEnhance.Color(img).enhance(1.35)
        img = ImageEnhance.Contrast(img).enhance(1.15)

    if keep_aspect:
        img.thumbnail(target_size, Image.LANCZOS)
        canvas = Image.new("RGB", target_size, (255, 255, 255))
        offset = ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2)
        canvas.paste(img, offset)
        img = canvas
    else:
        img = img.resize(target_size, Image.LANCZOS)

    return img


def pil_to_part(img: Image.Image) -> types.Part:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, subsampling=0)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def detect_dominant_colors(img: Image.Image) -> str:
    w, h = img.size
    cx, cy = w // 2, h // 2
    crop = img.crop((cx - w // 4, cy - h // 4, cx + w // 4, cy + h // 4))
    crop = crop.resize((50, 50), Image.LANCZOS)

    pixels = [p for p in list(crop.getdata())
              if not (p[0] > 230 and p[1] > 230 and p[2] > 230)]
    if not pixels:
        return "unknown color"

    r = sum(p[0] for p in pixels) // len(pixels)
    g = sum(p[1] for p in pixels) // len(pixels)
    b = sum(p[2] for p in pixels) // len(pixels)
    brightness = (r + g + b) / 3

    if brightness < 40:  return "pure black"
    if brightness < 75:  return "very dark / near-black"
    if brightness < 120:
        if b > r + 20 and b > g + 20: return "dark navy blue"
        if r > g + 20 and r > b + 20: return "dark red / maroon"
        return "dark charcoal"
    if r > g + 40 and r > b + 40: return "red"
    if g > r + 30 and g > b + 30: return "green"
    if b > r + 30 and b > g + 30: return "blue"
    if r > 200 and g > 180 and b < 80:  return "yellow"
    if r > 200 and g < 130 and b > 150: return "pink / magenta"
    if brightness > 210: return "white or off-white"
    return "medium neutral tone"


def build_prompt(transformation: str, top_color: str, bottom_color: str) -> str:
    body_instructions = {
        "none": (
            "BODY: Keep the person's body EXACTLY as it is in the photo. "
            "Do not alter their physique, weight, or muscle definition in any way."
        ),
        "1_month": (
            "BODY: Subtly enhance only the VISIBLE, UNCLOTHED parts of the body "
            "(face, neck, hands, forearms if bare) to reflect ~1 month of gym training. "
            "Show slightly reduced facial/neck fat and a hint of improved posture. "
            "Do NOT alter body shape under clothing — the clothes hide it. "
            "Keep changes minimal and photorealistic. Person must stay recognisable."
        ),
        "4_months": (
            "BODY: Enhance only the VISIBLE, UNCLOTHED parts of the body "
            "(face, neck, hands, forearms if bare) to reflect ~4 months of gym training. "
            "Show a noticeably leaner face/neck, sharper jawline, improved posture, "
            "and more defined forearms/hands where visible. "
            "Do NOT alter body shape under clothing — the clothes hide it. "
            "Keep it photorealistic. Person must stay clearly recognisable."
        ),
    }
    body_text = body_instructions.get(transformation, body_instructions["none"])

    return (
        "You are a high-end fashion retouching AI performing a virtual try-on.\n\n"

        "=== TASK ===\n"
        "You are given three images:\n"
        "  Image 1 — the person to dress (portrait)\n"
        "  Image 2 — the TOP garment to put on them\n"
        "  Image 3 — the BOTTOM garment to put on them\n\n"
        "Produce ONE output image: the person wearing both garments.\n\n"

        "=== COLOR ACCURACY (HIGHEST PRIORITY) ===\n"
        f"C1. The TOP (Image 2) is {top_color}. "
        f"Output MUST show it as {top_color}. Never lighten, gray, or desaturate it.\n"
        f"C2. The BOTTOM (Image 3) is {bottom_color}. "
        f"Output MUST show it as {bottom_color}. Never lighten, gray, or desaturate it.\n"
        "C3. Pure black → TRUE BLACK (RGB ≈ 0,0,0). Dark gray is wrong.\n"
        "C4. Navy → TRUE NAVY BLUE. Light blue or gray is wrong.\n"
        "C5. Fabric wrinkles/shading are fine as subtle surface detail only — "
        "they must never shift the perceived base color.\n\n"

        "=== CLOTHING SWAP RULES ===\n"
        "T1. The TOP garment must cover the ENTIRE upper body (shoulders to waist). "
        "No original clothing visible.\n"
        "T2. The BOTTOM garment must cover the ENTIRE lower body (waist to feet). "
        "No original clothing visible.\n"
        "T3. Both garments must be fully visible — nothing cropped at frame edges.\n"
        "T4. Keep the original background and lighting unchanged.\n"
        "T5. Add natural fabric draping, wrinkles, and fit for photorealism.\n\n"

        "=== FACE & IDENTITY ===\n"
        "F1. The person's face, hair, and skin tone are SACRED — do not alter them "
        "beyond what is explicitly requested in the BODY section below.\n\n"

        "=== BODY ===\n"
        f"{body_text}\n\n"

        "Output: a single photorealistic portrait image."
    )


def call_gemini_with_retry(
    user_img: Image.Image, top_img: Image.Image,
    bottom_img: Image.Image, prompt: str,
    max_retries: int = MAX_RETRIES,
):
    user_part   = pil_to_part(user_img)
    top_part    = pil_to_part(top_img)
    bottom_part = pil_to_part(bottom_img)
    last_error  = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"☁️  Gemini attempt {attempt}/{max_retries}...")
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_text(text="Image 1 — PORTRAIT (person to dress):"),
                        user_part,
                        types.Part.from_text(text="Image 2 — TOP GARMENT:"),
                        top_part,
                        types.Part.from_text(text="Image 3 — BOTTOM GARMENT:"),
                        bottom_part,
                        types.Part.from_text(text=prompt),
                    ])
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(aspect_ratio="9:16"),
                ),
            )

            if not response.candidates:
                raise ValueError("Empty candidates in response.")

            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    print(f"✅ Got image on attempt {attempt}.")
                    return part.inline_data

            raise ValueError("No inline_data image part in response.")

        except Exception as e:
            last_error = e
            print(f"⚠️  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    raise HTTPException(
        status_code=502,
        detail=f"Gemini failed after {max_retries} attempts. Last error: {last_error}",
    )


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


# ─── Main endpoint ────────────────────────────────────────────────────────────
@app.post("/api/process-transformation")
async def process_transformation(
    user_file:      UploadFile = File(...),
    top_id:         str        = Form(...),
    bottom_id:      str        = Form(...),
    transformation: str        = Form(...),
):
    # Validate transformation value
    if transformation not in ("none", "1_month", "4_months"):
        raise HTTPException(status_code=400, detail="Invalid transformation value.")

    # 1. Load — safe_catalog_path blocks directory traversal
    try:
        raw_user   = Image.open(io.BytesIO(await user_file.read()))
        raw_top    = Image.open(safe_catalog_path("tops",    top_id))
        raw_bottom = Image.open(safe_catalog_path("bottoms", bottom_id))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open image: {e}")

    # 2. Detect colors on raw pixels
    top_color    = detect_dominant_colors(raw_top    if raw_top.mode    == "RGB" else raw_top.convert("RGB"))
    bottom_color = detect_dominant_colors(raw_bottom if raw_bottom.mode == "RGB" else raw_bottom.convert("RGB"))
    print(f"🎨 top: [{top_color}]  bottom: [{bottom_color}]  transform: [{transformation}]")

    # 3. Normalize
    user_img   = normalize_image(raw_user,   TARGET_SIZE,   keep_aspect=True, boost_saturation=False)
    top_img    = normalize_image(raw_top,    CLOTHING_SIZE, keep_aspect=True, boost_saturation=True)
    bottom_img = normalize_image(raw_bottom, CLOTHING_SIZE, keep_aspect=True, boost_saturation=True)

    # 4. Prompt
    prompt = build_prompt(transformation, top_color, bottom_color)

    # 5. Call Gemini
    inline_data = call_gemini_with_retry(user_img, top_img, bottom_img, prompt)

    # 6. Stream back
    return StreamingResponse(io.BytesIO(inline_data.data), media_type=inline_data.mime_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
