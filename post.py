#!/usr/bin/env python3
import os, json, time, random, re
from datetime import datetime, timedelta
from io import BytesIO

from PIL import Image, ImageEnhance
import requests

# Try to enable Google Trends (free).
try:
    from pytrends.request import TrendReq
    HAVE_PYTRENDS = True
except Exception:
    HAVE_PYTRENDS = False

# ======================================================
# CONFIG
# ======================================================
OUT_DIR      = "out"
ASSETS_DIR   = "assets"
IMAGES_DIR   = os.path.join(ASSETS_DIR, "images")
STATE_DIR    = "state"

LIBRARY_PATH   = os.getenv("LIBRARY_PATH", f"{ASSETS_DIR}/veni_library.json")
HASHTAGS_PATH  = os.getenv("HASHTAGS_PATH", f"{ASSETS_DIR}/hashtags.json")
HISTORY_PATH   = os.getenv("HISTORY_PATH",  f"{STATE_DIR}/history.json")

# Overlay logo settings
LOGO_PATH         = os.getenv("LOGO_PATH", f"{IMAGES_DIR}/Logo.png")
LOGO_WIDTH_RATIO  = float(os.getenv("LOGO_WIDTH_RATIO", "0.18"))  # 18% of base width
LOGO_MARGIN_RATIO = float(os.getenv("LOGO_MARGIN_RATIO", "0.03")) # 3% margins
LOGO_OPACITY      = float(os.getenv("LOGO_OPACITY", "0.85"))      # 85% visible

# History / reuse window
NO_REPEAT_DAYS = int(os.getenv("NO_REPEAT_DAYS", "30"))

# IG Story (optional)
IG_USER_ID      = os.getenv("IG_USER_ID", "").strip()
IG_ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN", "").strip()
POST_STORY      = os.getenv("POST_STORY", "true").lower() == "true"
STORY_CAPTION   = os.getenv("STORY_CAPTION", "New post is live — tap profile").strip()

RUN_ID  = os.getenv("GITHUB_RUN_ID", str(int(time.time())))
RUN_TAG = "".join(ch for ch in RUN_ID if ch.isalnum())[-6:] or str(int(time.time()))[-6:]

# ======================================================
# UTIL
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

# ======================================================
# LIBRARY SELECTION
# { "file": "001.png", "quote": "...", "reflection": "...", "commentary": "..." }
# ======================================================
def is_recent(filename, history, days):
    now = datetime.utcnow()
    for h in reversed(history):
        if h.get("file") == filename:
            try:
                then = datetime.fromisoformat(h["ts"])
                if now - then < timedelta(days=days):
                    return True
            except Exception:
                pass
            break
    return False

def least_recent_filename(candidates, history):
    last_used = {h["file"]: h["ts"] for h in history if "file" in h and "ts" in h}
    def score(fn):
        ts = last_used.get(fn)
        if not ts:
            return datetime(1970,1,1)  # never used → oldest
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return datetime(1970,1,1)
    return sorted(candidates, key=lambda fn: score(fn))[0]

def pick_next_entry(library, history):
    def file_index(e):
        try:
            base = os.path.splitext(e.get("file",""))[0]
            return int(re.sub(r"[^0-9]", "", base) or "9999")
        except Exception:
            return 9999

    lib_sorted = sorted([e for e in library if e.get("file")], key=file_index)
    if not lib_sorted:
        return None

    eligible = [e for e in lib_sorted if not is_recent(e["file"], history, NO_REPEAT_DAYS)]

    if eligible:
        random.shuffle(eligible)
        return eligible[0]
    else:
        files = [e["file"] for e in lib_sorted]
        lr = least_recent_filename(files, history)
        for e in lib_sorted:
            if e["file"] == lr:
                return e
        return lib_sorted[0]

# ======================================================
# TONE BUCKET DETECTION
# 001–035 Ascetic Monk | 036–045 Disciplined Warrior
# 046–055 Serene Mystic | 056–065 Prophetic Visionary
# 066–100 Modern Pilgrim
# ======================================================
def tone_from_filename(fname: str) -> str:
    try:
        n = int(re.sub(r"[^0-9]", "", os.path.splitext(fname)[0]))
    except Exception:
        n = 1000
    if 1 <= n <= 35:   return "ascetic_monk"
    if 36 <= n <= 45:  return "disciplined_warrior"
    if 46 <= n <= 55:  return "serene_mystic"
    if 56 <= n <= 65:  return "prophetic_visionary"
    return "modern_pilgrim"

# ======================================================
# HASHTAGS
# ======================================================
def fetch_trending_hashtags(tone_bucket: str, max_tags=3, geo="ZA"):
    if not HAVE_PYTRENDS:
        return []
    seeds = {
        "ascetic_monk": ["monastic prayer", "contemplative spirituality", "silence and solitude"],
        "disciplined_warrior": ["armor of god", "faith over fear", "spiritual warfare"],
        "serene_mystic": ["christian mysticism", "god in nature", "sacred stillness"],
        "prophetic_visionary": ["revival", "justice and mercy", "prophetic voice"],
        "modern_pilgrim": ["walk by faith", "everyday faith", "christian journey"]
    }
    kw_list = seeds.get(tone_bucket, ["christian faith"])
    try:
        pytrends = TrendReq(hl="en-ZA", tz=120)
        pytrends.build_payload(kw_list, timeframe="now 7-d", geo=geo)
        related = pytrends.related_queries() or {}
        candidates = []
        for kw in kw_list:
            rising = (related.get(kw) or {}).get("rising")
            if rising is not None and "query" in rising:
                for q in list(rising["query"].head(10)):
                    tag = "#" + re.sub(r"[^a-z0-9]", "", q.lower().strip().replace(" ", ""))
                    if 3 <= len(tag) <= 40:
                        candidates.append(tag)
        deduped = list(dict.fromkeys(candidates))
        banned = {"#venifaithful", "#venipublishing"}
        return [t for t in deduped if t not in banned][:max_tags]
    except Exception:
        return []

def build_hashtag_line(tone_bucket: str):
    core = load_json(HASHTAGS_PATH, default={})
    branded = core.get("branded", ["#VeniFaithful"])
    general = core.get("general", [])
    tone_list = core.get(tone_bucket, [])

    random.shuffle(branded)
    random.shuffle(general)
    random.shuffle(tone_list)

    chosen_branded = branded[:2] if len(branded) >= 2 else branded
    chosen_core    = tone_list[:5] if len(tone_list) >= 5 else tone_list
    chosen_general = general[:2] if len(general) >= 2 else general
    trending       = fetch_trending_hashtags(tone_bucket, max_tags=3)

    final = []
    for lst in (chosen_branded, chosen_core, trending, chosen_general):
        for t in lst:
            if t not in final:
                final.append(t)

    return "" if not final else " ".join(final)

# ======================================================
# IMAGE SAVE (IG-safe JPEG) + OVERLAY
# ======================================================
def save_jpeg_ig_safe(img: Image.Image, path: str, max_side=1080, max_bytes=7_500_000):
    # Resize to IG-recommended bound
    w, h = img.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    # Adaptive quality loop until under size cap
    q = 92
    data = None
    while q >= 70:
        buf = BytesIO()
        img.save(buf, "JPEG", quality=q, optimize=True, progressive=True, subsampling=2)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            with open(path, "wb") as f:
                f.write(data)
            return
        q -= 4
    with open(path, "wb") as f:
        f.write(data)

def overlay_logo_once(input_path: str, output_path: str, logo_path=LOGO_PATH):
    """
    If output_path exists → do nothing.
    Else load input_path, overlay logo, save IG-safe JPEG to output_path.
    """
    if os.path.isfile(output_path):
        return

    try:
        base = Image.open(input_path).convert("RGB")
    except Exception as e:
        raise FileNotFoundError(f"Base image missing or unreadable: {input_path}") from e

    try:
        logo = Image.open(logo_path).convert("RGBA")
    except Exception:
        # No logo? still write a JPEG-safe version
        save_jpeg_ig_safe(base, output_path)
        return

    bw, bh = base.size
    target_w = max(1, int(bw * LOGO_WIDTH_RATIO))
    w, h = logo.size
    scale = target_w / float(w)
    logo = logo.resize((target_w, max(1, int(h * scale))), Image.LANCZOS)

    if LOGO_OPACITY < 1.0:
        alpha = logo.split()[-1]
        from PIL import ImageEnhance as IE
        alpha = IE.Brightness(alpha).enhance(LOGO_OPACITY)
        logo.putalpha(alpha)

    margin = int(bw * LOGO_MARGIN_RATIO)
    x = bw - logo.size[0] - margin
    y = bh - logo.size[1] - margin

    base_rgba = base.convert("RGBA")
    base_rgba.alpha_composite(logo, (x, y))
    final = base_rgba.convert("RGB")

    save_jpeg_ig_safe(final, output_path)

# ======================================================
# IG STORY (Content Publishing API supports story image posting; no link sticker)
# ======================================================
def ig_publish_story(ig_user_id: str, access_token: str, image_url: str, caption: str = "") -> str:
    # 1) Create a media container for a Story
    create_res = requests.post(
        f"https://graph.facebook.com/v21.0/{ig_user_id}/media",
        data={
            "image_url": image_url,
            "caption": caption,
            "media_type": "STORIES",
            "access_token": access_token,
        },
        timeout=30,
    )
    create_res.raise_for_status()
    creation_id = (create_res.json() or {}).get("id")
    if not creation_id:
        raise RuntimeError(f"Failed to create Story container: {create_res.text}")

    # 2) Quick poll
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

    # 3) Publish
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
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)

    library  = load_json(LIBRARY_PATH, default=[])
    hashtags = load_json(HASHTAGS_PATH, default={})
    history  = load_json(HISTORY_PATH, default=[])

    if not library:
        raise RuntimeError("No entries found. Ensure assets/veni_library.json exists and is valid.")

    # 1) Select entry with 30-day reuse rule
    entry = pick_next_entry(library, history)
    if not entry:
        raise RuntimeError("No selectable entry in library (check 'file' fields).")

    file        = (entry.get("file") or "").strip()
    quote       = (entry.get("quote") or "").strip()
    reflection  = (entry.get("reflection") or "").strip()
    commentary  = (entry.get("commentary") or "").strip()

    if not file or not quote:
        raise RuntimeError("Selected entry missing 'file' or 'quote'.")

    # 2) Source image path
    src_path = os.path.join(IMAGES_DIR, file)
    if not os.path.isfile(src_path):
        # try next available entry
        others = [e for e in library if e is not entry]
        picked = None
        for alt in others:
            p = os.path.join(IMAGES_DIR, alt.get("file",""))
            if os.path.isfile(p):
                picked = alt; src_path = p; break
        if not picked:
            raise FileNotFoundError(f"Image not found: {src_path}")
        entry       = picked
        file        = entry.get("file","")
        quote       = (entry.get("quote") or "").strip()
        reflection  = (entry.get("reflection") or "").strip()
        commentary  = (entry.get("commentary") or "").strip()

    # 3) Cached overlay output path: out/NNN-overlay.jpg
    base_name = os.path.splitext(os.path.basename(file))[0]  # 'NNN'
    out_name  = f"{base_name}-overlay.jpg"
    out_path  = os.path.join(OUT_DIR, out_name)

    # Create overlay once if not present (JPEG, IG-safe)
    overlay_logo_once(src_path, out_path, LOGO_PATH)

    # 4) Build caption (+ hashtags)
    tone_bucket = tone_from_filename(file)
    hashtag_line = build_hashtag_line(tone_bucket)
    hashtag_block = f"\n\n{hashtag_line}" if hashtag_line else ""

    caption = (
        f"{quote}\n\n"
        f"Reflection:\n{reflection}\n\n"
        f"Commentary:\n{commentary}"
        f"{hashtag_block}"
    )

    # 5) Public URL for the composited image
    repo = os.getenv("GITHUB_REPOSITORY", "") or "owner/repo"
    image_url = f"https://raw.githubusercontent.com/{repo}/main/{out_path.replace(os.sep, '/')}"

    # 6) Build payload for the downstream step that actually posts to Instagram
    payload = {
        "image_url": image_url,
        "caption": caption,
        "file": file,
        "tone": tone_bucket
    }
    with open("post_payload.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 7) Update history (keep lean)
    history.append({"file": file, "ts": datetime.utcnow().isoformat()})
    history = history[-400:]
    save_json(HISTORY_PATH, history)

    # 8) Optional: Story publish (uses same image URL; API cannot add a tappable link sticker)
    if POST_STORY and IG_USER_ID and IG_ACCESS_TOKEN:
        try:
            sid = ig_publish_story(IG_USER_ID, IG_ACCESS_TOKEN, image_url=image_url, caption=STORY_CAPTION)
            print(f"[story] Published Story ID: {sid}")
        except Exception as e:
            print(f"[story] Warning: Story publish failed: {e}")
    else:
        print("[story] Skipped (POST_STORY is false or IG creds not set)")

    print(f"✅ Ready to post: {out_path}\n\n{caption}\n")
