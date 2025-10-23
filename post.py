import os, json, uuid, base64, textwrap
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
TONE = os.getenv("TONE", "motivational, concise, modern")
STYLE = os.getenv("STYLE", "dark minimalist aesthetic, cinematic lighting")

HF_TOKEN = os.getenv("HF_TOKEN")
STABILITY_KEY = os.getenv("STABILITY_API_KEY")

if not HF_TOKEN:
    raise RuntimeError("Missing HF_TOKEN (Hugging Face token)")
if not STABILITY_KEY:
    raise RuntimeError("Missing STABILITY_API_KEY (Stability AI key)")

HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# -----------------------------
# HUGGING FACE TEXT GENERATION
# -----------------------------
def hf_complete(prompt, max_new_tokens=160):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.8,
            "return_full_text": False
        }
    }
    r = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers, json=payload, timeout=60
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    raise RuntimeError(f"Unexpected HF response: {str(data)[:300]}")

def generate_verse_and_reflection():
    verse_prompt = f"Write one short, powerful quote about {THEME}. It must be generic, ≤16 words, in quotation marks."
    reflection_prompt = f"Write a 1–2 sentence reflection expanding the quote for an Instagram audience. Tone: {TONE}. Style: {STYLE}. ≤40 words."
    verse = hf_complete(verse_prompt, 64)
    if not verse.startswith(("“", "\"")):
        verse = "“" + verse.strip('"').strip('“').strip() + "”"
    reflection = hf_complete(reflection_prompt, 120)
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

    verse, reflection = generate_verse_and_reflection()

    img_bytes = generate_image_bytes(
        f"{THEME}, {STYLE}, instagram composition, detailed, high quality"
    )
    base_img = Image.open(BytesIO(img_bytes)).convert("RGB")

    final_img = draw_centered_text(base_img, verse, reflection) if OVERLAY_TEXT_ON_IMAGE else base_img

    today = datetime.utcnow().strftime("%Y-%m-%d")
    filename = f"{today}.jpg"
    out_path = os.path.join(OUT_DIR, filename)
    final_img.save(out_path, "JPEG", quality=95)

    hashtags = "#Discipline #Focus #Perseverance #DailyMotivation"
    caption = f"{verse}\n\n{reflection}\n\n{hashtags}"

    repo = os.getenv("GITHUB_REPOSITORY")
    image_url = f"https://raw.githubusercontent.com/{repo}/main/{out_path.replace(os.sep, '/')}"

    payload = {"image_url": image_url, "caption": caption}
    with open("post_payload.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {out_path}\n{caption}")
