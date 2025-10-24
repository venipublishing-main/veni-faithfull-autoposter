import os, json, base64, textwrap
from datetime import datetime
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# -----------------------------
# CONFIGURATION
# -----------------------------
OUT_DIR = "out"
FONT_PATH = "assets/fonts/IMFellEnglishSC-Regular.ttf"
OVERLAY_TEXT_ON_IMAGE = True  # set False to post clean image without text overlay

# You can override these by setting GitHub env vars
THEME = os.getenv("THEME", "discipline, focus, perseverance")
TONE  = os.getenv("TONE",  "motivational, concise, modern")
STYLE = os.getenv("STYLE", "dark minimalist aesthetic, cinematic lighting")

GEMINI_KEY     = os.getenv("GEMINI_API_KEY")
STABILITY_KEY  = os.getenv("STABILITY_API_KEY")

if not GEMINI_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")
if not STABILITY_KEY:
    raise RuntimeError("Missing STABILITY_API_KEY")

# -----------------------------
# GEMINI TEXT GENERATION (free tier)
# -----------------------------
# Model name kept as env for easy swaps; default to a Flash model suitable for free tier
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

def gemini_generate(prompt: str) -> str:
    """
    Calls the new Google GenAI endpoint.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.8, "maxOutputTokens": 200}
    }
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        raise RuntimeError(f"Unexpected Gemini response: {str(data)[:300]}")

def generate_verse_and_reflection():
    prompt = (
        f"Generate TWO short pieces of text about {THEME}.\n"
        f"1. A brief, powerful quote (≤16 words, in quotation marks, no attribution).\n"
        f"2. A concise 1–2 sentence reflection (≤40 words) expanding on the quote.\n"
        f"Tone: {TONE}. Style: {STYLE}."
    )
    full_text = gemini_generate(prompt)
    parts = full_text.split("\n", 1)
    verse = parts[0].strip()
    reflection = parts[1].strip() if len(parts) > 1 else ""
    return verse, reflection

# -----------------------------
# STABILITY IMAGE GENERATION
# -----------------------------
def generate_image_bytes(prompt, width=1024, height=1024):
    url = "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image"
    headers = {"Authorization": f"Bearer {STABILITY_KEY}"}
    body = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "width": width,
        "height": height,
        "samples": 1
    }
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    return base64.b64decode(r.json()["artifacts"][0]["base64"])

# -----------------------------
# TEXT OVERLAY
# -----------------------------
def draw_centered_text(img, verse, reflection):
    W, H = img.size
    draw = ImageDraw.Draw(img)
    try:
        font_verse = ImageFont.truetype(FONT_PATH, 64)
        font_reflection = ImageFont.truetype(FONT_PATH, 36)
    except Exception:
        font_verse = font_reflection = ImageFont.load_default()

    verse_wrapped = textwrap.fill(verse, width=28)
    reflection_wrapped = textwrap.fill(reflection, width=38)

    y = int(H * 0.16)
    for line in verse_wrapped.split("\n"):
        w, h = draw.textsize(line, font=font_verse)
        draw.text(((W - w) / 2, y), line, font=font_verse, fill=(255, 255, 255))
        y += h + 8

    y += 40
    for line in reflection_wrapped.split("\n"):
        w, h = draw.textsize(line, font=font_reflection)
        draw.text(((W - w) / 2, y), line, font=font_reflection, fill=(235, 235, 235))
        y += h + 6

    return img

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Text via Gemini
    verse, reflection = generate_verse_and_reflection()

    # 2) Image via Stability
    img_bytes = generate_image_bytes(
        f"{THEME}, {STYLE}, instagram composition, detailed, high quality"
    )
    base_img = Image.open(BytesIO(img_bytes)).convert("RGB")

    # 3) Optional overlay
    final_img = draw_centered_text(base_img, verse, reflection) if OVERLAY_TEXT_ON_IMAGE else base_img

    # 4) Save
    today = datetime.utcnow().strftime("%Y-%m-%d")
    filename = f"{today}.jpg"
    out_path = os.path.join(OUT_DIR, filename)
    final_img.save(out_path, "JPEG", quality=95)

    # 5) Caption + payload
    hashtags = "#Discipline #Focus #Perseverance #DailyMotivation"
    caption = f"{verse}\n\n{reflection}\n\n{hashtags}"

    repo = os.getenv("GITHUB_REPOSITORY")
    image_url = f"https://raw.githubusercontent.com/{repo}/main/{out_path.replace(os.sep, '/')}"

    payload = {"image_url": image_url, "caption": caption}
    with open("post_payload.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {out_path}\n{caption}")
