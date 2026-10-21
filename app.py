import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
from google import genai
from google.genai import types
import time
import re

# ============================================================
#  CONFIG — Model name এখানে change করবেন ভবিষ্যতে
# ============================================================
GEMINI_MODEL = "gemini-3.5-flash-lite"
# ভবিষ্যতে "gemini-3.5-flash-lite" আসলে শুধু এটুকু change করুন

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PDF OCR — Gemini AI",
    page_icon="📄",
    layout="wide",
)

# ============================================================
#  CSS
# ============================================================
st.markdown("""
<style>
    .hero {
        text-align: center;
        padding: 1.5rem 1rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; color: #fff; }
    .hero p  { margin: .4rem 0 0; color: rgba(255,255,255,.85); }

    .card {
        background: #1a1d2e;
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: #fff;
        padding: .25rem .9rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: .85rem;
    }
    .ok  { border-left: 4px solid #4CAF50; background: #122a1a; padding: .7rem 1rem; border-radius: 0 8px 8px 0; color: #81C784; }
    .warn{ border-left: 4px solid #FFC107; background: #2a2510; padding: .7rem 1rem; border-radius: 0 8px 8px 0; color: #FFD54F; }
    .err { border-left: 4px solid #f44336; background: #2a1212; padding: .7rem 1rem; border-radius: 0 8px 8px 0; color: #E57373; }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  HELPERS
# ============================================================

def parse_page_input(raw: str, total: int) -> list[int]:
    """
    User input parse করে valid page list বানায়।
    Supports:
        "3"          → [3]
        "1,3,5"      → [1,3,5]
        "2-6"        → [2,3,4,5,6]
        "1,3,7-10"   → [1,3,7,8,9,10]
    """
    pages = set()
    parts = raw.replace(" ", "").split(",")
    for part in parts:
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                a, b = int(a), int(b)
                if a > b:
                    a, b = b, a
                for p in range(a, b + 1):
                    if 1 <= p <= total:
                        pages.add(p)
            except ValueError:
                continue
        else:
            try:
                p = int(part)
                if 1 <= p <= total:
                    pages.add(p)
            except ValueError:
                continue
    return sorted(pages)


def render_single_page(pdf_bytes: bytes, page_num: int, dpi: int = 200):
    """শুধু একটা page render করে image bytes দেয় — FAST!"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_num - 1)          # 0-indexed
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


def ocr_page(client, img_bytes: bytes, page_num: int, lang: str) -> str:
    """একটা page এর image → Gemini দিয়ে OCR"""
    prompt = f"""You are an expert OCR engine. This is page {page_num} of a PDF.

RULES:
1. Extract ALL text from this page exactly as it appears.
2. Preserve formatting: headings, paragraphs, bullet points, numbering, tables.
3. Keep original language (Bengali stays Bengali, English stays English).
4. If there are tables, reproduce them in markdown table format.
5. If there are math formulas, write them clearly.
6. Output ONLY the extracted text — no explanation, no commentary.
{f"7. Primary language of this document: {lang}." if lang else ""}"""

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,
            ),
        )
        return resp.text.strip()
    except Exception as e:
        return f"❌ Error on page {page_num}: {e}"


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        placeholder="AIzaSy...",
        help="https://aistudio.google.com/apikey",
    )

    lang = st.selectbox(
        "🌐 Document Language",
        ["Auto Detect", "বাংলা", "English", "বাংলা + English", "Hindi", "Arabic"],
    )

    dpi = st.slider("🔍 DPI (Quality)", 100, 400, 200, 50,
                    help="বেশি = ভালো quality, কিন্তু একটু ধীর")

    st.markdown("---")
    st.markdown(f"""
    <div class="card">
        <b>Model:</b> <code>{GEMINI_MODEL}</code><br>
        <b>Speed:</b> ⚡ Fast<br>
        <b>Languages:</b> 🇧🇩 🇬🇧 Multi
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <b>📋 Page Input Format:</b><br>
        <code>3</code> → single page<br>
        <code>1,3,5</code> → multiple<br>
        <code>2-8</code> → range<br>
        <code>1,3,7-10</code> → mixed
    </div>
    """, unsafe_allow_html=True)

# ============================================================
#  HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>📄 PDF OCR — Page First, Then Extract</h1>
    <p>Page number দিন → শুধু সেই page OCR হবে → দ্রুত ও সহজ!</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
#  STEP 1 — UPLOAD
# ============================================================
st.markdown("### ① PDF আপলোড করুন")
uploaded = st.file_uploader("PDF ফাইল select করুন", type=["pdf"], label_visibility="collapsed")

if not uploaded:
    st.info("👆 উপরে PDF আপলোড করলে পরের step গুলো দেখা যাবে।")
    st.stop()

pdf_bytes = uploaded.getvalue()
doc = fitz.open(stream=pdf_bytes, filetype="pdf")
total_pages = len(doc)
doc.close()

st.markdown(f"""
<div class="ok">
    ✅ <b>{uploaded.name}</b> — {total_pages} pages, {len(pdf_bytes)/1024/1024:.2f} MB
</div>
""", unsafe_allow_html=True)

# ============================================================
#  STEP 2 — PAGE NUMBER INPUT  (এটাই মূল পরিবর্তন!)
# ============================================================
st.markdown("### ② Page Number দিন")

page_input = st.text_input(
    "কোন কোন page চান?  (e.g.  `1,3,5-8`)",
    placeholder="1, 3, 7-10",
    help="Comma দিয়ে আলাদা করুন, range দিতে hyphen ব্যবহার করুন",
)

if not page_input.strip():
    st.warning("⬆️ Page number লিখুন, তারপর OCR হবে।")
    st.stop()

selected = parse_page_input(page_input, total_pages)

if not selected:
    st.markdown(f'<div class="err">❌ কোনো valid page পাওয়া যায়নি। 1 থেকে {total_pages} এর মধ্যে দিন।</div>',
                unsafe_allow_html=True)
    st.stop()

st.markdown(f"""
<div class="ok">
    📑 Selected: <b>{', '.join(map(str, selected))}</b>
    &nbsp;({len(selected)} টি page — শুধু এগুলোই render ও OCR হবে ⚡)
</div>
""", unsafe_allow_html=True)

# ============================================================
#  STEP 3 — PREVIEW THUMBNAILS (optional, fast — low DPI)
# ============================================================
with st.expander(f"👁️ Preview ({len(selected)} pages)", expanded=False):
    cols = st.columns(min(len(selected), 6))
    for i, pg in enumerate(selected[:12]):
        thumb_bytes = render_single_page(pdf_bytes, pg, dpi=90)  # low DPI = fast
        with cols[i % 6]:
            st.image(thumb_bytes, caption=f"Page {pg}", use_container_width=True)
    if len(selected) > 12:
        st.caption(f"... আরো {len(selected)-12} টি page")

# ============================================================
#  STEP 4 — OCR BUTTON
# ============================================================
st.markdown("---")

if not api_key:
    st.markdown('<div class="warn">⚠️ Sidebar এ API Key দিন!</div>', unsafe_allow_html=True)
    st.stop()

run_ocr = st.button(
    f"🚀 OCR করুন — {len(selected)} page",
    type="primary",
    use_container_width=True,
)

# ============================================================
#  STEP 5 — RUN OCR (only selected pages!)
# ============================================================
if run_ocr:
    client = genai.Client(api_key=api_key)
    lang_hint = "" if lang == "Auto Detect" else lang

    results: dict[int, str] = {}
    bar = st.progress(0, text="Starting...")

    for idx, pg in enumerate(selected):
        bar.progress(
            (idx) / len(selected),
            text=f"⏳ Page {pg} render + OCR হচ্ছে... ({idx+1}/{len(selected)})",
        )

        # 1) শুধু এই page টা render (বাকি গুলো touch ও করি না!)
        img_bytes = render_single_page(pdf_bytes, pg, dpi=dpi)

        # 2) OCR
        text = ocr_page(client, img_bytes, pg, lang_hint)
        results[pg] = text

        # rate-limit safety
        if idx < len(selected) - 1:
            time.sleep(0.8)

    bar.progress(1.0, text="✅ সব done!")
    st.session_state["ocr_results"] = results
    st.session_state["pdf_name"] = uploaded.name

# ============================================================
#  STEP 6 — RESULTS
# ============================================================
if "ocr_results" in st.session_state:
    results = st.session_state["ocr_results"]

    st.markdown("---")
    st.markdown("## 📝 OCR Results")

    # --- Combined text ---
    combined = "\n\n".join(
        f"{'='*50}\n📄 PAGE {pg}\n{'='*50}\n\n{results[pg]}"
        for pg in sorted(results)
    )

    st.markdown("### 📋 সব একসাথে (Copy করুন)")
    st.text_area(
        "Full Text",
        value=combined,
        height=220,
        key="full_text",
        label_visibility="collapsed",
    )

    # --- Per-page results ---
    st.markdown("### 📄 Page অনুযায়ী")

    for pg in sorted(results):
        with st.expander(f"📄 Page {pg}", expanded=True):
            c1, c2 = st.columns([1, 2])

            with c1:
                thumb = render_single_page(pdf_bytes, pg, dpi=120)
                st.image(thumb, caption=f"Page {pg}", use_container_width=True)

            with c2:
                st.markdown(f'<span class="badge">Page {pg}</span>', unsafe_allow_html=True)
                st.text_area(
                    f"Text {pg}",
                    value=results[pg],
                    height=350,
                    key=f"pg_{pg}",
                    label_visibility="collapsed",
                )
                wc = len(results[pg].split())
                cc = len(results[pg])
                st.caption(f"Words: {wc} · Chars: {cc}")

    # --- Downloads ---
    st.markdown("---")
    dl1, dl2 = st.columns(2)
    fname = st.session_state.get("pdf_name", "output").replace(".pdf", "")

    with dl1:
        st.download_button(
            "📥 Download TXT",
            data=combined,
            file_name=f"ocr_{fname}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📥 Download MD",
            data=combined,
            file_name=f"ocr_{fname}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # Clear
    if st.button("🗑️ Clear Results", use_container_width=True):
        del st.session_state["ocr_results"]
        st.rerun()