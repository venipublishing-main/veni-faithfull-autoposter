#!/usr/bin/env python3
import os, json, base64, textwrap, math, random, time, io, re
from datetime import datetime, timedelta
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from io import BytesIO

# Optional CV libs (required for AI-aware placement)
import numpy as np
import cv2

# ======================================================
# PER-RUN UNIQUENESS SEEDING
# ======================================================
RUN_ID = os.getenv("GITHUB_RUN_ID", str(int(time.time())))
RUN_TAG = RUN_ID[-6:]  # for unique filenames
try:
    _seed_digits = "".join(ch for ch in RUN_ID if ch.isdigit())
    random.seed(int(_seed_digits[-9:]) if _seed_digits else int(time.time()))
except Exception:
    random.seed(int(time.time()))

# ======================================================
# CONFIGURATION
# ======================================================
OUT_DIR = "out"
ASSETS_DIR = "assets"
STATE_DIR = "state"
LIBRARY_PATH = os.getenv("LIBRARY_PATH", f"{ASSETS_DIR}/veni_library.json")
HISTORY_PATH = f"{STATE_DIR}/history.json"

# --- New free-mode toggles ---
IMAGE_MODE = os.getenv("IMAGE_MODE", "auto").lower()          # auto | typography
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "stability")      # reserved for future multi-provider use
FALLBACK_FONT_PATH = os.getenv("FALLBACK_FONT_PATH", "assets/fonts/IMFellEnglishSC-Regular.ttf")

FONT_PATH = "assets/fonts/IMFellEnglishSC-Regular.ttf"
OVERLAY_TEXT_ON_IMAGE = True  # toggle to post clean image if False

# Env defaults
THEME = os.getenv("THEME", "discipline, focus, perseverance").lower()
TONE  = os.getenv("TONE",  "motivational, concise, modern")
ATTRIBUTION_MODE = os.getenv("ATTRIBUTION_MODE", "hybrid").lower()  # hybrid | ai
BIBLE_TRANSLATION = os.getenv("BIBLE_TRANSLATION", "auto").upper()  # KJV | WEB | AUTO
NO_REPEAT_DAYS = int(os.getenv("NO_REPEAT_DAYS", "21"))

GEMINI_KEY        = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "400"))
STABILITY_KEY     = os.getenv("STABILITY_API_KEY", "").strip()
STABILITY_ENGINE  = os.getenv("STABILITY_ENGINE", "stable-diffusion-xl-1024-v1-0").strip()

# === Instagram Story publishing (optional) ===
IG_USER_ID      = os.getenv("IG_USER_ID", "").strip()
IG_ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN", "").strip()
POST_STORY      = os.getenv("POST_STORY", "false").lower() == "true"
STORY_CAPTION   = os.getenv("STORY_CAPTION", "New post is live — tap profile").strip()

if not GEMINI_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")
# NOTE: STABILITY_API_KEY is OPTIONAL now (free local fallback covers $0 mode)

# ======================================================
# STYLE FAMILIES (theme → visual aesthetic)
# ======================================================
STYLE_FAMILIES = {
    "ascetic": (
        "monastic minimalism, aged parchment texture, muted warm light, "
        "meditative atmosphere, candlelight shadows, classical composition"
    ),
    "warrior": (
        "cinematic chiaroscuro, metal and stone texture, dramatic contrast, "
        "embers and dust, determination, dynamic lighting"
    ),
    "contemplative": (
        "soft focus, natural light through fog, tranquil balance, "
        "subtle color harmony, zen-like simplicity"
    ),
    "visionary": (
        "ethereal glow, sunrise tones, clarity and optimism, "
        "spacious composition, bright volumetric light, hopeful atmosphere"
    ),
    "mystical": (
        "deep indigo palette, star-lit ambience, sacred geometry hints, "
        "symbolic surrealism, dreamlike textures"
    ),
}

def pick_style_from_theme(theme: str) -> str:
    if "discipline" in theme or "ascetic" in theme:
        return STYLE_FAMILIES["ascetic"]
    if "focus" in theme or "perseverance" in theme or "warrior" in theme:
        return STYLE_FAMILIES["warrior"]
    if "reflection" in theme or "mind" in theme or "thought" in theme or "peace" in theme:
        return STYLE_FAMILIES["contemplative"]
    if "vision" in theme or "light" in theme or "clarity" in theme or "hope" in theme:
        return STYLE_FAMILIES["visionary"]
    return STYLE_FAMILIES["mystical"]

# ======================================================
# LIBRARY + HISTORY (Hybrid mode)
# ======================================================
def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def theme_tags(theme_str: str):
    return [t.strip().lower() for t in theme_str.split(",") if t.strip()]

def filter_entries_by_tags(entries, tags):
    if not tags:
        return entries[:]
    out = []
    tagset = set(tags)
    for e in entries:
        etags = set((e.get("tags") or []))
        if etags & tagset:
            out.append(e)
    return out or entries[:]  # fallback if none matched

def is_recent(id_, history, days):
    now = datetime.utcnow()
    for item in history:
        if item.get("id") == id_:
            try:
                dt = datetime.fromisoformat(item.get("ts"))
                if now - dt < timedelta(days=days):
                    return True
            except Exception:
                continue
    return False

# ======================================================
# GEMINI (Reflection + Closer)
# ======================================================
def _count_sentences(text: str) -> int:
    return sum(1 for s in re.split(r"[.!?]+", text) if s.strip())

def gemini_reflection(text_for_context: str, ref_or_author: str, kind: str) -> str:
    """
    Aim for 2–3 sentences, about 60–100 words total. One retry if too short.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    sys = (
        "You write contemplative Instagram reflections. "
        "Output 2 to 3 sentences total, about 60–100 words. "
        "Do NOT restate or quote the provided text. Avoid hashtags and emojis."
    )
    def _call(temp=0.7):
        usr = (
            f"Kind: {kind}\n"
            f"Source: {ref_or_author}\n"
            f"Text:\n{text_for_context}\n\n"
            f"Write a reflection for modern readers. Warm, clear, and practical."
        )
        payload = {
            "systemInstruction": {"role": "system", "parts": [{"text": sys}]},
            "contents": [{"role": "user", "parts": [{"text": usr}]}],
            "generationConfig": {"temperature": temp, "maxOutputTokens": GEMINI_MAX_TOKENS}
        }
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        txt = (
            (data.get("candidates") or [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        ).strip()
        return txt

    txt = _call(0.7)
    if _count_sentences(txt) < 2:
        txt = _call(0.6)  # one gentle retry

    words = txt.split()
    if len(words) > 110:
        txt = " ".join(words[:110]).rstrip(".!,;:") + "."
    return txt or "Let this truth shape your next step today."

def gemini_closer(text_for_context: str, kind: str) -> str:
    """
    One short closing line (<= 12 words) that invites contemplation or action.
    No hashtags/emojis. Returns empty string on failure.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    sys = (
        "You write a single short closing line for an Instagram caption. "
        "At most 12 words. No hashtags or emojis."
    )
    usr = (
        f"Kind: {kind}\n"
        f"Text:\n{text_for_context}\n\n"
        f"Write ONE short closing line that invites contemplation or gentle action."
    )
    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": sys}]},
        "contents": [{"role": "user", "parts": [{"text": usr}]}],
        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 40}
    }
    try:
        r = requests.post(url, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()
        line = (
            (data.get("candidates") or [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        ).strip()
        words = line.split()
        if len(words) > 12:
            line = " ".join(words[:12]).rstrip(".!,;:") + "."
        if len(line) < 3:
            return ""
        return line
    except Exception:
        return ""

# ======================================================
# FREE LOCAL BACKGROUND (no external API)
# ======================================================
def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

def _wrap_to_width(draw, text, font, max_w):
    lines, cur = [], []
    for word in text.split():
        test = (" ".join(cur + [word])).strip()
        w = draw.textlength(test, font=font)
        if w <= max_w or not cur:
            cur.append(word)
        else:
            lines.append(" ".join(cur)); cur = [word]
    if cur:
        lines.append(" ".join(cur))
    return lines

def _auto_fit_block(draw, text, target_w, target_h, max_size, min_size=18, step=2, font_path=FALLBACK_FONT_PATH):
    size = max_size
    while size >= min_size:
        font = _load_font(font_path, size)
        lines = _wrap_to_width(draw, text, font, target_w)
        if not lines:
            size -= step; continue
        line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
        total_h = int(len(lines) * (line_h + math.ceil(size * 0.25)))
        max_line_w = max(draw.textlength(l, font=font) for l in lines)
        if total_h <= target_h and max_line_w <= target_w:
            return font, lines, line_h
        size -= step
    font = _load_font(font_path, min_size)
    lines = _wrap_to_width(draw, text, font, target_w)
    line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
    return font, lines, line_h

def _apply_vignette(im, strength=0.4):
    W, H = im.size
    vign = Image.new("L", (W, H), 0)
    vdraw = ImageDraw.Draw(vign)
    vdraw.ellipse([-W*0.2, -H*0.2, W*1.2, H*1.2], fill=int(220*strength))
    vign = vign.filter(ImageFilter.GaussianBlur(int(min(W, H)*0.12)))
    im = Image.composite(im, Image.new("RGB", (W,H), (0,0,0)), vign)
    return im

def _free_background(width=1024, height=1024):
    # Subtle radial gradient + noise; theme nudges hue
    base = Image.new("RGB", (width, height), (22, 24, 32))
    grad = Image.new("L", (width, height), 0)
    g = ImageDraw.Draw(grad)
    cx, cy = width//2, height//2
    maxr = int((width**2 + height**2) ** 0.5)//2
    for r in range(0, maxr, 8):
        val = min(255, int(60 + r*0.6))
        g.ellipse([cx-r, cy-r, cx+r, cy+r], fill=val)
    grad = grad.filter(ImageFilter.GaussianBlur(45))
    tint = (34, 36, 48)
    if "warrior" in THEME or "persever" in THEME:
        tint = (38, 34, 36)
    elif "vision" in THEME or "hope" in THEME:
        tint = (40, 44, 54)
    elif "mystic" in THEME or "indigo" in THEME:
        tint = (28, 30, 44)
    bg = Image.composite(Image.new("RGB", (width,height), tint), base, grad)
    # subtle noise
    noise = Image.effect_noise((width,height), 8).convert("L")
    bg = Image.blend(bg, Image.merge("RGB",(noise,noise,noise)), 0.08)
    # vignette
    bg = _apply_vignette(bg, strength=0.35)
    return bg

# ======================================================
# STABILITY IMAGE GENERATION (optional, with fallback)
# ======================================================
def _stability_generate(prompt, width=1024, height=1024, seed=None):
    engines = [STABILITY_ENGINE]  # keep single known-good; deprecated engines removed
    negative = (
        "text, letters, typography, captions, watermark, logo, signature, words, "
        "flat plain background, low detail, low contrast, artifacts"
    )
    headers = {"Authorization": f"Bearer {STABILITY_KEY}", "Accept": "application/json"}
    last_err = None
    for engine in engines:
        url = f"https://api.stability.ai/v1/generation/{engine}/text-to-image"
        seed_value = seed if isinstance(seed, int) else random.randint(1, 2_147_483_000)
        body = {
            "text_prompts": [
                {"text": f"{prompt}, no text, no typography, clean background"},
                {"text": negative, "weight": -1.2}
            ],
            "cfg_scale": 7,
            "width": width,
            "height": height,
            "samples": 1,
            "seed": seed_value,
        }
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code == 404:
                print(f"[Stability] 404 for engine '{engine}' — skipping.")
                continue
            r.raise_for_status()
            data = r.json()
            return base64.b64decode(data["artifacts"][0]["base64"])
        except Exception as e:
            last_err = e
            try:
                print(f"[Stability] Engine '{engine}' failed: {r.status_code} {r.text[:200]}")
            except Exception:
                print(f"[Stability] Engine '{engine}' failed: {e}")
    if last_err:
        raise last_err
    raise RuntimeError("Stability generation failed without explicit error.")

def generate_image_bytes(prompt, width=1024, height=1024, seed=None):
    """
    Returns JPEG bytes for the base image.
    - If IMAGE_MODE == 'typography' OR no STABILITY_KEY: free local background.
    - Else try Stability once; on any error, fall back to free local background.
    """
    # Free mode or no API key → local background
    if IMAGE_MODE == "typography" or not STABILITY_KEY:
        print("[image] Using FREE local background (typography mode or no Stability key).")
        bg = _free_background(width, height)
        out = BytesIO()
        bg.save(out, "JPEG", quality=92)
        out.seek(0)
        return out.read()

    # Paid attempt (if configured)
    try:
        img = _stability_generate(prompt, width=width, height=height, seed=seed)
        return img
    except Exception as e:
        print(f"[image] Stability failed ({e}); falling back to FREE local background.")
        bg = _free_background(width, height)
        out = BytesIO()
        bg.save(out, "JPEG", quality=92)
        out.seek(0)
        return out.read()

# ======================================================
# AI-AWARE LAYOUT (saliency + brightness + golden ratio)
# ======================================================
def _saliency_like(gray_u8: np.ndarray) -> np.ndarray:
    """
    Fast saliency-ish map using Laplacian + Sobel energy, smoothed and normalized to [0,1].
    """
    lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
    gx  = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    sob = cv2.magnitude(gx, gy)
    sal = 0.6 * np.abs(lap) + 0.4 * sob
    sal = cv2.GaussianBlur(sal, (0, 0), 3)
    sal -= sal.min()
    if sal.max() > 1e-6:
        sal /= sal.max()
    return sal

def _brightness(gray_u8: np.ndarray) -> np.ndarray:
    return gray_u8.astype(np.float32) / 255.0

def _score_region(sal_map: np.ndarray, bright_map: np.ndarray, rect):
    x1, y1, x2, y2 = rect
    roi_sal = sal_map[y1:y2, x1:x2]
    roi_br  = bright_map[y1:y2, x1:x2]
    if roi_sal.size == 0:
        return 1e9
    mean_sal = float(roi_sal.mean())
    std_sal  = float(roi_sal.std())
    mean_br  = float(roi_br.mean())
    penalty = 0.0
    if mean_br < 0.25:
        penalty += (0.25 - mean_br) * 0.6
    if mean_br > 0.90:
        penalty += (mean_br - 0.90) * 0.6
    return 0.7 * mean_sal + 0.3 * std_sal + penalty

def _find_best_text_box(pil_img, box_rel=(0.78, 0.36), margin_rel=0.08):
    w, h = pil_img.size
    img_rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sal = _saliency_like(gray)
    br  = _brightness(gray)

    margin = int(w * margin_rel)
    bw = max(int(w * box_rel[0]), 100)
    bh = max(int(h * box_rel[1]), 100)

    xs = [margin, (w - bw)//2, max(w - bw - margin, margin)]
    ys = [margin, (h - bh)//2, max(h - bh - margin, margin)]
    candidates = [(xx, yy, xx + bw, yy + bh) for yy in ys for xx in xs]

    # Golden ratio candidates
    gx = int((w - bw) * 0.382); gy = int((h - bh) * 0.382)
    candidates += [(gx, gy, gx + bw, gy + bh),
                   (w - bw - gx, h - bh - gy, w - gx, h - gy)]

    best_rect, best_score = None, 1e9
    for rect in candidates:
        score = _score_region(sal, br, rect)
        if score < best_score:
            best_score, best_rect = score, rect

    # Clamp
    x1, y1, x2, y2 = best_rect
    x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
    x2 = max(x1+1, min(x2, w)); y2 = max(y1+1, min(y2, h))
    return (x1, y1, x2, y2)

# ======================================================
# TYPOGRAPHY & RENDER
# ======================================================
def _apply_bottom_gradient(img, strength=220):
    w, h = img.size
    grad = Image.new("L", (1, h), 0)
    for i in range(h):
        t = max(0, (i - int(h*0.55)) / (h*0.45))
        val = int((t**1.8) * strength)
        grad.putpixel((0, i), val)
    alpha = grad.resize((w, h))
    black = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    black.putalpha(alpha)
    return Image.alpha_composite(img.convert("RGBA"), black).convert("RGB")

def _pick_text_colors(img, rect):
    x1, y1, x2, y2 = rect
    region = img.crop((x1, y1, x2, y2)).resize((1,1), Image.BOX)
    r, g, b = region.getpixel((0,0))
    luma = 0.2126*r + 0.7152*g + 0.0722*b
    if luma < 110:
        fill = (245, 245, 245)   # light text
        shadow = (0, 0, 0, 120)
    else:
        fill = (25, 25, 25)
        shadow = (0, 0, 0, 80)
    return fill, shadow

def _rounded_backdrop(mask_size, rect, radius=40, alpha=66):
    w, h = mask_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = rect
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=alpha)
    mask = mask.filter(ImageFilter.GaussianBlur(3.0))  # softer feather
    return mask

def draw_quote_and_attrib(img, quote_text, attribution):
    img = _apply_bottom_gradient(img)
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    verse_box = _find_best_text_box(img, box_rel=(0.78, 0.36), margin_rel=0.08)
    fill, shadow = _pick_text_colors(img, verse_box)

    # Adaptive backdrop opacity based on local brightness
    x1, y1, x2, y2 = verse_box
    region = img.crop((x1, y1, x2, y2)).convert("L").resize((1,1), Image.BOX)
    avg = region.getpixel((0,0)) / 255.0
    if avg > 0.75:
        backdrop_alpha = 96
    elif avg < 0.35:
        backdrop_alpha = 54
    else:
        backdrop_alpha = 66

    # Typography hierarchy
    q_max = int(w * 0.085)
    a_ratio = 0.75  # attribution smaller
    q_font, q_lines, q_line_h = _auto_fit_block(
        draw, quote_text, target_w=verse_box[2]-verse_box[0], target_h=int((verse_box[3]-verse_box[1]) * 0.78),
        max_size=q_max, font_path=FONT_PATH
    )
    attrib_max = max(18, int(q_font.size * a_ratio))
    a_font, a_lines, a_line_h = _auto_fit_block(
        draw, attribution, target_w=verse_box[2]-verse_box[0], target_h=int((verse_box[3]-verse_box[1]) * 0.22),
        max_size=attrib_max, font_path=FONT_PATH
    )

    # Backdrop under the combined block
    q_block_h = int(len(q_lines) * (q_line_h + math.ceil(q_font.size * 0.25)))
    a_block_h = int(len(a_lines) * (a_line_h + math.ceil(a_font.size * 0.25)))
    pad = 20
    rect = (verse_box[0], verse_box[1], verse_box[0] + int(max(
        max(draw.textlength(l, font=q_font) for l in q_lines) if q_lines else 0,
        max(draw.textlength(l, font=a_font) for l in a_lines) if a_lines else 0
    )) + pad*2, verse_box[1] + q_block_h + a_block_h + pad*3)

    mask = _rounded_backdrop((w, h), rect, radius=40, alpha=backdrop_alpha)
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    overlay.putalpha(mask)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # Draw text with soft shadow
    x = rect[0] + pad
    y = rect[1] + pad
    for line in q_lines:
        draw.text((x+2, y+2), line, font=q_font, fill=shadow)
        draw.text((x, y), line, font=q_font, fill=fill)
        y += q_line_h + math.ceil(q_font.size * 0.25)
    y += pad // 2
    attrib_vis = attribution.replace(" ", "  ")
    for line in _wrap_to_width(draw, attrib_vis, a_font, rect[2]-rect[0]-pad*2):
        draw.text((x+1, y+1), line, font=a_font, fill=(0,0,0,90))
        draw.text((x, y), line, font=a_font, fill=tuple(int(c*0.9) for c in fill[:3]))
        y += a_line_h + math.ceil(a_font.size * 0.25)

    return img

# ======================================================
# IG STORY PUBLISH HELPERS
# ======================================================
def ig_publish_story(ig_user_id: str, access_token: str, image_url: str, caption: str = "") -> str:
    """
    Create an Instagram Story container from a public image URL and publish it.
    Returns the Story media ID on success. Raises on failure.
    """
    # 1) Create a media container for a Story
    create_res = requests.post(
        f"https://graph.facebook.com/v21.0/{ig_user_id}/media",
        data={
            "image_url": image_url,
            "caption": caption,
            "media_type": "STORIES",  # Stories via Content Publishing API
            "access_token": access_token,
        },
        timeout=30,
    )
    create_res.raise_for_status()
    creation_id = (create_res.json() or {}).get("id")
    if not creation_id:
        raise RuntimeError(f"Failed to create Story container: {create_res.text}")

    # 2) (Optional) brief poll; images are usually instant
    for _ in range(8):
        status = requests.get(
            f"https://graph.facebook.com/v21.0/{creation_id}",
            params={"fields": "status", "access_token": access_token},
            timeout=15,
        )
        status.raise_for_status()
        if (status.json() or {}).get("status") == "FINISHED":
            break
        time.sleep(1)

    # 3) Publish the Story
    pub_res = requests.post(
        f"https://graph.facebook.com/v21.0/{ig_user_id}/media_publish",
        data={"creation_id": creation_id, "access_token": access_token},
        timeout=30,
    )
    pub_res.raise_for_status()
    story_id = (pub_res.json() or {}).get("id")
    if not story_id:
        raise RuntimeError(f"Failed to publish Story: {pub_res.text}")
    return story_id

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)

    # 0) Load library + history
    library = load_json(LIBRARY_PATH, default=[])
    history = load_json(HISTORY_PATH, default=[])

    # 1) Select entry (Hybrid) or error if library missing
    if ATTRIBUTION_MODE == "hybrid" and library:
        # Build candidate pool (so we can pick an alternate if needed)
        tags = theme_tags(THEME)
        pool = filter_entries_by_tags(library, tags)
        overlay_pool = [e for e in pool if e.get("allowed_overlay", True)]
        if overlay_pool:
            pool = overlay_pool
        pool = [e for e in pool if not is_recent(e.get("id"), history, NO_REPEAT_DAYS)]
        if not pool:
            pool = library[:]

        random.shuffle(pool)
        entry = pool[0]
        last_id = history[-1]["id"] if history else None
        if last_id and entry.get("id") == last_id and len(pool) > 1:
            entry = pool[1]

        # EARLY history write (prevents same selection on quick reruns)
        history.append({"id": entry.get("id"), "ts": datetime.utcnow().isoformat()})
        save_json(HISTORY_PATH, history)

        kind = entry.get("kind", "bible")

        if kind == "bible":
            ref = entry.get("ref", "")
            trans = entry.get("translations", {})
            t_pref = BIBLE_TRANSLATION
            text = None
            if t_pref == "KJV":
                text = trans.get("KJV")
            elif t_pref == "WEB":
                text = trans.get("WEB")
            else:  # AUTO
                text = trans.get("WEB") if random.random() < 0.5 and trans.get("WEB") else trans.get("KJV")
            if not text:
                text = trans.get("KJV") or trans.get("WEB") or ""
            quote_text = text.strip()
            attribution = f"— {ref} ({'WEB' if text == trans.get('WEB') and trans.get('WEB') else 'KJV'})"

        elif kind == "historical":
            quote_text = (entry.get("text") or "").strip()
            attribution = f"— {entry.get('author','Unknown')}".strip()

        else:  # original
            quote_text = (entry.get("text") or "").strip()
            attribution = "— Inspired Reflection"

        # guardrail: if overlay too long, push quote to caption only (still generate image)
        words = quote_text.split()
        overlay_ok = entry.get("allowed_overlay", True) and len(words) <= 32

        # 2) Reflection via Gemini
        reflection = gemini_reflection(quote_text, attribution.lstrip("— ").strip(), kind)

        # 3) Style (theme-mapped, allow style_hint override)
        style = pick_style_from_theme(THEME)
        if entry.get("style_hint"):
            style = STYLE_FAMILIES.get(entry["style_hint"], style)

        # 4) Image generation (paid if available → free fallback)
        hint = " ".join(quote_text.split()[:6])
        img_prompt = (
            f"Concept art inspired by themes of {THEME}; "
            f"visualize the feeling of '{hint}' without any written text. "
            f"Visual style: {style}. High-contrast, richly textured, "
            f"cinematic depth of field, professional Instagram composition."
        )
        try:
            base_seed = int(("".join(ch for ch in RUN_ID if ch.isdigit()))[-9:])
        except Exception:
            base_seed = random.randint(1, 2_147_483_000)
        img_seed = (base_seed + random.randint(1, 100_000)) % 2_147_483_000

        img_bytes = generate_image_bytes(img_prompt, seed=img_seed)
        base_img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # 5) Overlay (quote + attribution) or clean image
        if OVERLAY_TEXT_ON_IMAGE and overlay_ok:
            final_img = draw_quote_and_attrib(base_img, "“" + quote_text.strip("“”\"") + "”", attribution)
        else:
            final_img = base_img

        # 6) Save output (unique filename per run)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        filename = f"{today}-{RUN_TAG}.jpg"
        out_path = os.path.join(OUT_DIR, filename)
        final_img.save(out_path, "JPEG", quality=95)

        # 7) Caption: Quote → Reflection → Closer
        hashtags = "#Discipline #Focus #Perseverance #DailyMotivation"
        closer = gemini_closer(quote_text, kind)
        closer_block = ("\n" + closer) if closer else ""
        # Ensure reflection is present and has at least 2 sentences
        if not reflection or len(reflection.strip()) < 40 or _count_sentences(reflection) < 2:
            reflection = (
                "Let this guide your next step: turn intention into practice, "
                "practice into character, and character into quiet strength."
            )

        caption = (
            "“" + quote_text.strip("“”\"") + "”\n"
            + attribution + "\n\n"
            + reflection
            + closer_block + "\n\n"
            + hashtags
        )

        # 8) Payload
        repo = os.getenv("GITHUB_REPOSITORY", "") or "owner/repo"
        image_url = f"https://raw.githubusercontent.com/{repo}/main/{out_path.replace(os.sep, '/')}"
        payload = {"image_url": image_url, "caption": caption}
        with open("post_payload.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # 8.5) Optional: Publish a Story using the same image URL
        if POST_STORY and IG_USER_ID and IG_ACCESS_TOKEN:
            try:
                story_id = ig_publish_story(
                    ig_user_id=IG_USER_ID,
                    access_token=IG_ACCESS_TOKEN,
                    image_url=image_url,      # reuse the feed image URL
                    caption=STORY_CAPTION
                )
                print(f"[story] Published Story ID: {story_id}")
            except Exception as e:
                print(f"[story] Warning: Story publish failed: {e}")
        else:
            print("[story] Skipped (POST_STORY is false or IG creds not set)")

        # 9) Prune history (older than 60 days)
        cutoff = datetime.utcnow() - timedelta(days=60)
        history = [h for h in history if datetime.fromisoformat(h["ts"]) >= cutoff]
        save_json(HISTORY_PATH, history)

        print(f"✅ Generated {out_path}\n{caption}")

    else:
        raise RuntimeError(
            "Hybrid mode requested but library not found or empty. "
            "Ensure assets/veni_library.json exists with entries."
        )
