import os, io, csv, random, textwrap, datetime, json, pathlib
from dataclasses import dataclass
from typing import Optional, Tuple
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# -------- CONFIG --------
BRAND_NAME = "Veni Faithfull"
OUTPUT_DIR = "out"
IMG_W, IMG_H = 1080, 1350       # Portrait works well on IG
MARGIN = 80
FONT_HEAD = None  # will load default if None
FONT_BODY = None
FONT_FOOT = None
HASHTAGS = "#VerseOfTheDay #Faith #Hope #Scripture #Christian"

# Pull from env in GitHub Actions
IG_TOKEN   = os.getenv("IG_TOKEN", "").strip()
IG_USER_ID = os.getenv("IG_USER_ID", "").strip()
REPO       = os.getenv("GITHUB_REPOSITORY", "yourname/yourrepo")
BRANCH     = os.getenv("GITHUB_REF_NAME", "main") or "main"  # e.g. main

# Bible API endpoints (free); fallback local list if needed
BIBLE_API = "https://bible-api.com/{}?translation=kjv"
FALLBACK_VERSES = [
    ("Joshua 1:9", "Have not I commanded thee? Be strong and of a good courage; be not afraid, neither be thou dismayed: for the LORD thy God is with thee whithersoever thou goest."),
    ("Psalm 23:1", "The LORD is my shepherd; I shall not want."),
    ("Proverbs 3:5-6", "Trust in the LORD with all thine heart; and lean not unto thine own understanding. In all thy ways acknowledge him, and he shall direct thy paths."),
    ("Philippians 4:6", "Be careful for nothing; but in every thing by prayer and supplication with thanksgiving let your requests be made known unto God."),
]

@dataclass
class Content:
    kind: str  # "bible" or "saint"
    title: str # "John 3:16" or "Augustine"
    text: str  # verse or quote
    source_url: Optional[str] = None

def today_filename() -> str:
    d = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"{d}.jpg"

def choose_content(repo_root: str) -> Content:
    # 60% Bible, 40% Saint
    if random.random() < 0.6:
        return get_bible_content()
    else:
        return get_saint_content(repo_root)

def get_bible_content() -> Content:
    # Pick a random fallback ref or construct random ref
    ref = random.choice([
        "john 3:16", "psalm 91:1", "romans 8:28", "isaiah 40:31",
        "matthew 6:33", "philippians 4:13", "proverbs 3:5-6",
        "psalm 23:1", "joshua 1:9", "1 corinthians 13:4-7"
    ])
    try:
        r = requests.get(BIBLE_API.format(ref), timeout=10)
        if r.ok:
            j = r.json()
            text = j.get("text", "").strip()
            reference = j.get("reference", ref.title())
            if text:
                return Content("bible", reference, text)
    except Exception:
        pass
    # Fallback local
    reference, verse = random.choice(FALLBACK_VERSES)
    return Content("bible", reference, verse)

def get_saint_content(repo_root: str) -> Content:
    path = pathlib.Path(repo_root) / "data" / "saints_quotes.csv"
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        row = random.choice(rows)
        author = row["author"].strip()
        quote  = row["quote"].strip()
        src    = (row.get("source_url") or "").strip() or None
        title  = author
        return Content("saint", title, quote, src)
    except Exception:
        # Fallback if file missing
        return Content("saint", "Augustine",
                       "Our hearts are restless until they rest in You, O Lord.",
                       "https://www.ccel.org/ccel/augustine/confessions.txt")

# --- Reflection generator (template-based; ~40–60 words) ---
OPENERS_BIBLE = [
    "Today’s verse invites us to lean into God’s presence.",
    "This passage reminds us that faith steadies a restless heart.",
    "Scripture here calls us to take courage and move with trust.",
]
INSIGHTS_BIBLE = [
    "Courage isn’t the absence of fear, but confidence that God walks with us.",
    "Hope grows when we surrender outcomes and obey the next small step.",
    "Peace deepens as we choose prayer over worry, one decision at a time.",
]
APPLY_BIBLE = [
    "Speak this verse aloud and carry it into one task today.",
    "Pray briefly now, then act on the good you already know to do.",
    "Encourage one person who might need this word.",
]

OPENERS_SAINT = [
    "This insight from {who} points our hearts toward deeper freedom.",
    "With {who}, we learn that grace matures through daily choices.",
    "{who} reminds us that love orders our desires toward God.",
]
INSIGHTS_SAINT = [
    "Holiness grows when we return to God with honesty and hope.",
    "True strength is humble; it chooses truth over image.",
    "Joy slowly rises where gratitude and patience are practiced.",
]
APPLY_SAINT = [
    "Practice one concrete act of quiet charity today.",
    "Pause for a minute of silence; let God speak in the stillness.",
    "Share this with someone who needs encouragement.",
]

def generate_reflection(c: Content) -> str:
    if c.kind == "bible":
        parts = [
            random.choice(OPENERS_BIBLE),
            random.choice(INSIGHTS_BIBLE),
            random.choice(APPLY_BIBLE),
        ]
    else:
        parts = [
            random.choice([p.format(who=c.title) for p in OPENERS_SAINT]),
            random.choice(INSIGHTS_SAINT),
            random.choice(APPLY_SAINT),
        ]
    txt = " ".join(parts)
    # keep ~60 words max
    words = txt.split()
    if len(words) > 70:
        txt = " ".join(words[:70]) + "…"
    return txt

# --- Image rendering ---
def load_font(size: int) -> ImageFont.FreeTypeFont:
    # Use bundled DejaVu if available in runner; fallback to default bitmap font
    try:
        return ImageFont.truetype("DejaVuSerif.ttf", size)
    except:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except:
            return ImageFont.load_default()

def draw_wrapped(draw, text, font, x, y, max_w, line_h, fill):
    # wrap by words to fit max_w
    for line in textwrap.wrap(text, width=40):
        w, h = draw.textbbox((0,0), line, font=font)[2:]
        # if too wide, reduce width by splitting more aggressively
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y

def render_image(c: Content, brand: str, save_path: str):
    # Background: soft parchment
    img = Image.new("RGB", (IMG_W, IMG_H), (245, 239, 229))
    d = ImageDraw.Draw(img)

    # Subtle vignette
    vignette = Image.new("L", (IMG_W, IMG_H), 0)
    dv = ImageDraw.Draw(vignette)
    dv.ellipse((-200, -100, IMG_W+200, IMG_H+300), fill=255)
    vignette = vignette.filter(ImageFilter.GaussianBlur(120))
    img = Image.composite(img, Image.new("RGB", img.size, (235, 228, 216)), vignette)

    # Cross watermark
    cross = Image.new("L", (IMG_W, IMG_H), 0)
    dc = ImageDraw.Draw(cross)
    cx, cy = IMG_W//2, IMG_H//3
    dc.rectangle((cx-12, cy-220, cx+12, cy+220), fill=40)
    dc.rectangle((cx-130, cy-12, cx+130, cy+12), fill=40)
    cross = cross.filter(ImageFilter.GaussianBlur(4))
    img = Image.composite(img, Image.new("RGB", img.size, (220, 213, 200)), cross)

    # Text blocks
    title_font = load_font(60)
    body_font  = load_font(48)
    foot_font  = load_font(36)

    # Title
    y = MARGIN
    title = c.title if c.kind == "saint" else f"{c.title}"
    tw, th = d.textbbox((0,0), title, font=title_font)[2:]
    d.text(((IMG_W - tw)//2, y), title, font=title_font, fill=(30,30,30))
    y += th + 30

    # Quote/Verse
    wrap_width = 36 if c.kind == "bible" else 34
    # manual wrap using textwrap
    lines = textwrap.wrap(c.text.strip(), width=wrap_width)
    # draw each line centered
    for line in lines:
        lw, lh = d.textbbox((0,0), line, font=body_font)[2:]
        d.text(((IMG_W - lw)//2, y), line, font=body_font, fill=(35,35,35))
        y += lh + 6

    # Footer brand
    foot = brand
    fw, fh = d.textbbox((0,0), foot, font=foot_font)[2:]
    d.text(((IMG_W - fw)//2, IMG_H - fh - MARGIN), foot, font=foot_font, fill=(60,60,60))

    # Save
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    img.save(save_path, "JPEG", quality=92, optimize=True)

def build_caption(c: Content, reflection: str) -> str:
    if c.kind == "bible":
        header = f"{c.text.strip()} — {c.title}"
    else:
        header = f"“{c.text.strip()}” — {c.title}"
    parts = [header, "", reflection, "", HASHTAGS]
    return "\n".join(parts).strip()

def raw_github_url(repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"

def post_to_instagram(image_url: str, caption: str) -> Tuple[str, str]:
    # Create container
    create_url = f"https://graph.facebook.com/v24.0/{IG_USER_ID}/media"
    r = requests.post(create_url, data={
        "image_url": image_url,
        "caption": caption,
        "access_token": IG_TOKEN
    }, timeout=30)
    r.raise_for_status()
    creation_id = r.json()["id"]

    # Publish
    publish_url = f"https://graph.facebook.com/v24.0/{IG_USER_ID}/media_publish"
    r2 = requests.post(publish_url, data={
        "creation_id": creation_id,
        "access_token": IG_TOKEN
    }, timeout=30)
    r2.raise_for_status()
    media_id = r2.json()["id"]
    return creation_id, media_id

def main():
    assert IG_TOKEN and IG_USER_ID, "Missing IG_TOKEN or IG_USER_ID env vars."

    repo_root = str(pathlib.Path(__file__).resolve().parent)
    c = choose_content(repo_root)
    reflection = generate_reflection(c)

    filename = today_filename()
    save_path = os.path.join(OUTPUT_DIR, filename)
    render_image(c, BRAND_NAME, save_path)

    # Build caption
    caption = build_caption(c, reflection)

    # Build the public raw URL for the image (served by GitHub)
    image_url = raw_github_url(REPO, BRANCH, f"{OUTPUT_DIR}/{filename}")

    # Try posting
    print(f"[INFO] Posting: {c.kind} | {c.title}")
    print(f"[INFO] Image URL: {image_url}")
    print(f"[INFO] Caption (first 80 chars): {caption[:80]!r}")

    # Important: the image must be committed & available at that URL BEFORE posting.
    # In GitHub Actions we commit & push after running this script, so for a single-run test:
    # 1) First run to generate & commit image, 2) Second run to publish.
    # To simplify, we’ll attempt a small retry loop to give time after commit step.
    # In the workflow, we'll run: (a) generate & commit, (b) wait, (c) publish.

    # For local/manual runs you can comment out posting lines until the image is online.

    # Return data for workflow via a small file
    with open("post_payload.json", "w", encoding="utf-8") as f:
        json.dump({"image_url": image_url, "caption": caption}, f)

if __name__ == "__main__":
    main()
