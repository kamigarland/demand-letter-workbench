import os
import re
import json
import time
import base64
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
import pdfplumber
import docx2txt
import fitz  # PyMuPDF
from docx import Document
from openai import OpenAI


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Demand Letter Workbench", layout="wide")
st.title("First-Party Property Demand Letter Workbench")

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not api_key:
    st.error("Missing OPENAI_API_KEY. Add it to Streamlit secrets (cloud) or environment variable (local).")
    st.stop()

client = OpenAI(api_key=api_key)

BEST_MODEL = "gpt-5.2"   # drafting + extraction
VISION_MODEL = "gpt-4.1-mini" # OCR + receipts

# Safety caps (adjust as needed)
DEFAULT_OCR_MAX_PAGES_PER_PDF = 12
DEFAULT_POLICY_CONTEXT_CHARS = 12000
DEFAULT_DOC_SNIPPET_CHARS = 12000


# =========================
# SESSION STATE
# =========================
def init_state():
    if "docs" not in st.session_state:
        st.session_state.docs = []  # list of dicts: {name, ext, kind, text, notes, ocr_used, pages_ocrd, meta}
    if "case_packet" not in st.session_state:
        st.session_state.case_packet = None  # dict
    if "draft" not in st.session_state:
        st.session_state.draft = ""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "case_id" not in st.session_state:
        st.session_state.case_id = f"case_{int(time.time())}"

init_state()


# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:
    st.header("Inputs")
    insured_name = st.text_input("Insured Name")
    claim_number = st.text_input("Claim Number")
    policy_number = st.text_input("Policy Number")
    jurisdiction = st.text_input("Jurisdiction (State)")
    deductible = st.number_input("Deductible ($)")
    prior_payments = st.number_input("Prior Payments ($)")
    tone = st.selectbox("Tone", ["Professional", "Firm", "Aggressive Litigation-Ready"])
    enable_pdf_ocr = st.toggle("OCR scanned PDFs", value=True)
    ocr_max_pages = st.number_input("Max OCR pages per PDF", min_value=1, max_value=60, value=12)
    pdf_render_zoom = st.slider("PDF OCR zoom", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    image_mode = st.selectbox("Images are…", ["Receipts / invoices (extract totals)", "Damage photos (describe damage)"])
    policy_context_chars = st.number_input("Policy excerpt size (chars)", min_value=3000, max_value=40000, value=12000)
    doc_snippet_chars = st.number_input("Doc snippet size (chars)", min_value=3000, max_value=40000, value=12000)
   
# =========================
# HELPERS
# =========================
def require_client():
    if client is None:
        st.error("OpenAI client not initialized.")
        st.stop()


def safe_json_load(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def guess_doc_kind(filename: str, first_text: str = "") -> str:
    name = filename.lower()
    t = (first_text or "").lower()

    # filename heuristics
    if "policy" in name or "declarations" in name or "endorsement" in name:
        return "policy"
    if "denial" in name or "coverage position" in name or "reservation" in name:
        return "denial"
    if "estimate" in name or "xactimate" in name or "scope" in name:
        return "estimate"
    if "proof of loss" in name or "spol" in name or "sworn" in name:
        return "spol"
    if "invoice" in name or "receipt" in name:
        return "receipt"
    if "claim" in name and "notes" in name:
        return "claim_notes"
    if "email" in name or "correspondence" in name:
        return "correspondence"

    # text heuristics
    if "loss payment" in t or "duties after loss" in t or "exclusions" in t or "coverage a" in t:
        return "policy"
    if "we are denying" in t or "we must respectfully deny" in t or "coverage is denied" in t:
        return "denial"
    if "rcv" in t and "acv" in t and ("line item" in t or "quantity" in t):
        return "estimate"
    if "sworn proof of loss" in t:
        return "spol"

    return "other"


def pdf_to_png_bytes_list(pdf_path: str, zoom: float, max_pages: int) -> List[bytes]:
    images: List[bytes] = []
    doc = fitz.open(pdf_path)
    try:
        mat = fitz.Matrix(zoom, zoom)
        page_count = min(doc.page_count, max_pages)
        for i in range(page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            images.append(pix.tobytes("png"))
    finally:
        doc.close()
    return images


def openai_vision_ocr_image_bytes(img_bytes: bytes, mime: str = "image/png") -> str:
    require_client()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    resp = client.responses.create(
        model=VISION_MODEL,
        instructions=(
            "Extract ALL text visible in the image exactly as written. "
            "Return plain text only. No commentary."
        ),
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "OCR this image."},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )
    return (resp.output_text or "").strip()


def extract_text_pdf_openai_ocr(pdf_path: str, zoom: float, max_pages: int) -> str:
    page_imgs = pdf_to_png_bytes_list(pdf_path, zoom=zoom, max_pages=max_pages)
    out = []
    for idx, img_bytes in enumerate(page_imgs, start=1):
        try:
            page_text = openai_vision_ocr_image_bytes(img_bytes, mime="image/png")
            out.append(f"[PAGE {idx}]\n{page_text}".strip())
        except Exception as e:
            out.append(f"[PAGE {idx}]\n[OCR failed: {e}]")
    return "\n\n".join(out).strip()


def ai_extract_receipt_json(img_bytes: bytes, mime: str) -> str:
    require_client()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    instructions = (
        "You are extracting data from a receipt/invoice image for an insurance claim file. "
        "Only extract what is actually visible. If a field is not visible, use null. "
        "Return STRICT JSON only (no markdown) with this schema:\n"
        "{"
        "\"merchant\": string|null, "
        "\"date\": string|null, "
        "\"total\": number|null, "
        "\"currency\": string|null, "
        "\"category_guess\": string|null, "
        "\"line_items\": [ {\"description\": string, \"amount\": number|null} ] | null, "
        "\"notes\": string|null"
        "}"
    )

    resp = client.responses.create(
        model=VISION_MODEL,
        instructions=instructions,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Extract receipt fields."},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )
    return (resp.output_text or "").strip()


def ai_describe_damage_photo(img_bytes: bytes, mime: str) -> str:
    require_client()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    resp = client.responses.create(
        model=VISION_MODEL,
        instructions=(
            "Describe only objectively visible property damage in 4–8 bullets. "
            "Do not speculate about cause. Do not invent measurements. "
            "If the image is not damage-related, say so."
        ),
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Summarize damage visible in the photo."},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )
    return (resp.output_text or "").strip()


def extract_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """
    Returns dict:
    {name, ext, kind, text, notes, ocr_used, pages_ocrd, meta}
    """
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()
    meta = {}

    # PDFs
    if ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            extracted = ""
            parse_error = None

            try:
                with pdfplumber.open(tmp_path) as pdf:
                    for page in pdf.pages:
                        extracted += (page.extract_text() or "") + "\n"
                extracted = extracted.strip()
                meta["pages"] = len(pdf.pages) if "pdf" in locals() else None
            except Exception as e:
                parse_error = str(e)
                extracted = ""

            # if enough text, keep it
            if len(extracted) >= 200:
                kind = guess_doc_kind(name, extracted[:2000])
                return {
                    "name": name,
                    "ext": ext,
                    "kind": kind,
                    "text": extracted,
                    "notes": "Text extracted (pdfplumber).",
                    "ocr_used": False,
                    "pages_ocrd": 0,
                    "meta": meta,
                }

            # fallback OCR
            if enable_pdf_ocr:
                try:
                    ocr_text = extract_text_pdf_openai_ocr(
                        pdf_path=tmp_path,
                        zoom=float(pdf_render_zoom),
                        max_pages=int(ocr_max_pages),
                    )
                    kind = guess_doc_kind(name, ocr_text[:2000])
                    return {
                        "name": name,
                        "ext": ext,
                        "kind": kind,
                        "text": ocr_text,
                        "notes": f"OCR fallback used. Parse error: {parse_error}" if parse_error else "OCR fallback used (scanned/empty PDF).",
                        "ocr_used": True,
                        "pages_ocrd": int(ocr_max_pages),
                        "meta": meta,
                    }
                except Exception as e:
                    kind = guess_doc_kind(name, "")
                    return {
                        "name": name,
                        "ext": ext,
                        "kind": kind,
                        "text": "",
                        "notes": f"PDF parse weak and OCR failed: {e}",
                        "ocr_used": True,
                        "pages_ocrd": 0,
                        "meta": meta,
                    }

            # no OCR
            kind = guess_doc_kind(name, extracted[:2000])
            return {
                "name": name,
                "ext": ext,
                "kind": kind,
                "text": extracted,
                "notes": f"PDF text empty/weak. Parse error: {parse_error}" if parse_error else "PDF text empty/weak; OCR disabled.",
                "ocr_used": False,
                "pages_ocrd": 0,
                "meta": meta,
            }

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # DOCX
    if ext == ".docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            text = (docx2txt.process(tmp_path) or "").strip()
            kind = guess_doc_kind(name, text[:2000])
            return {
                "name": name,
                "ext": ext,
                "kind": kind,
                "text": text,
                "notes": "Text extracted (docx2txt).",
                "ocr_used": False,
                "pages_ocrd": 0,
                "meta": meta,
            }
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # TXT / CSV
    if ext in [".txt", ".csv"]:
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        kind = guess_doc_kind(name, text[:2000])
        return {
            "name": name,
            "ext": ext,
            "kind": kind,
            "text": text,
            "notes": "Text loaded.",
            "ocr_used": False,
            "pages_ocrd": 0,
            "meta": meta,
        }

    # Images
    if ext in [".png", ".jpg", ".jpeg", ".webp"]:
        require_client()
        img_bytes = uploaded_file.getvalue()
        mime = "image/png" if ext == ".png" else "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/webp"

        try:
            if image_mode.startswith("Receipts"):
                receipt_json = ai_extract_receipt_json(img_bytes, mime)
                kind = "receipt_image"
                return {
                    "name": name,
                    "ext": ext,
                    "kind": kind,
                    "text": receipt_json,
                    "notes": "Receipt JSON extracted via vision.",
                    "ocr_used": True,
                    "pages_ocrd": 1,
                    "meta": meta,
                }
            else:
                desc = ai_describe_damage_photo(img_bytes, mime)
                kind = "damage_photo"
                return {
                    "name": name,
                    "ext": ext,
                    "kind": kind,
                    "text": desc,
                    "notes": "Damage description extracted via vision.",
                    "ocr_used": True,
                    "pages_ocrd": 1,
                    "meta": meta,
                }
        except Exception as e:
            return {
                "name": name,
                "ext": ext,
                "kind": "image",
                "text": "",
                "notes": f"Image vision extraction failed: {e}",
                "ocr_used": True,
                "pages_ocrd": 0,
                "meta": meta,
            }

    # unsupported
    return {
        "name": name,
        "ext": ext,
        "kind": "unsupported",
        "text": "",
        "notes": "Unsupported file type. Export to PDF/DOCX/TXT if needed.",
        "ocr_used": False,
        "pages_ocrd": 0,
        "meta": meta,
    }


# =========================
# POLICY CLAUSE RETRIEVAL (no embeddings)
# =========================
POLICY_KEYWORDS = [
    "Loss Payment", "Appraisal", "Duties After Loss", "Exclusions", "Perils Insured Against",
    "Wear and tear", "marring", "deterioration", "latent defect", "rust", "corrosion",
    "seepage", "leakage", "water damage", "mold", "fungi", "ordinance", "law",
    "additional living expense", "ALE", "reasonable repairs", "protect the property"
]

def extract_relevant_policy_excerpts(policy_text: str, denial_text: str, max_chars: int) -> str:
    """
    Keyword-window retrieval: finds keyword occurrences and returns surrounding context.
    Also uses denial_text keywords if available.
    """
    text = policy_text or ""
    if not text.strip():
        return ""

    # build keyword set
    kws = set(POLICY_KEYWORDS)
    # add denial-derived keywords (simple)
    denial_words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", (denial_text or ""))
    for w in denial_words[:200]:
        if w.lower() in ["policy", "insured", "claim", "damage", "property"]:
            continue
        if len(w) >= 5:
            kws.add(w)

    # find windows
    windows = []
    lower = text.lower()
    for kw in list(kws)[:250]:
        k = kw.lower()
        idx = 0
        while True:
            pos = lower.find(k, idx)
            if pos == -1:
                break
            start = max(0, pos - 800)
            end = min(len(text), pos + 1200)
            windows.append(text[start:end])
            idx = pos + len(k)
            if len(windows) >= 30:
                break
        if len(windows) >= 30:
            break

    # Deduplicate-ish
    seen = set()
    chunks = []
    for w in windows:
        sig = hash(w[:200])
        if sig in seen:
            continue
        seen.add(sig)
        chunks.append(w.strip())

    out = "\n\n---\n\n".join(chunks)
    return out[:max_chars]


# =========================
# CASE PACKET BUILDER
# =========================
def build_case_packet(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    require_client()

    # separate docs by kind
    denials = [d for d in docs if d["kind"] == "denial"]
    policies = [d for d in docs if d["kind"] == "policy"]
    estimates = [d for d in docs if d["kind"] == "estimate"]
    spols = [d for d in docs if d["kind"] == "spol"]
    receipts = [d for d in docs if d["kind"] in ["receipt", "receipt_image"]]
    claim_notes = [d for d in docs if d["kind"] in ["claim_notes", "correspondence", "other"]]
    damage_photos = [d for d in docs if d["kind"] == "damage_photo"]

    denial_text = "\n\n".join([d["text"] for d in denials if d["text"]])[:doc_snippet_chars]
    estimate_text = "\n\n".join([d["text"] for d in estimates if d["text"]])[:doc_snippet_chars]
    spol_text = "\n\n".join([d["text"] for d in spols if d["text"]])[:doc_snippet_chars]
    notes_text = "\n\n".join([d["text"] for d in claim_notes if d["text"]])[:doc_snippet_chars]
    receipts_text = "\n\n".join([d["text"] for d in receipts if d["text"]])[:doc_snippet_chars]
    photos_text = "\n\n".join([d["text"] for d in damage_photos if d["text"]])[:doc_snippet_chars]

    # policy excerpts: retrieve relevant windows, not full policy
    policy_combined = "\n\n".join([p["text"] for p in policies if p["text"]])
    policy_excerpts = extract_relevant_policy_excerpts(policy_combined, denial_text, max_chars=int(policy_context_chars))

    packet_prompt = f"""
You are a senior first-party property insurance coverage attorney. Build a STRICT JSON Case Fact Packet using ONLY the provided text.
Do not invent facts, dates, statutes, or policy language.

Return STRICT JSON only with keys:
{{
  "insured_name": string|null,
  "insured_property_address": string|null,
  "carrier_name": string|null,
  "adjuster_name": string|null,
  "adjuster_email": string|null,
  "claim_number": string|null,
  "policy_number": string|null,
  "date_of_loss": string|null,
  "loss_description": string|null,
  "timeline": [{{"date": string|null, "event": string, "source": string}}],
  "denial_reasons_verbatim": [string],
  "policy_excerpts_verbatim": [string],
  "damages": {{
      "highest_estimate_rcv": number|null,
      "highest_estimate_acv": number|null,
      "depreciation": number|null,
      "notes": string|null,
      "prior payments": string|null
  }},
  "receipts": {{
      "items": [{{"merchant": string|null, "date": string|null, "total": number|null, "category_guess": string|null, "notes": string|null, "source": string}}],
      "total_visible": number|null
  }},
  "missing_inputs": [string],
  "top_leverage_points": [string]
}}

INPUTS:
Manual Overrides (may be null):
- insured_name: {insured_name or "null"}
- claim_number: {claim_number or "null"}
- policy_number: {policy_number or "null"}
- deductible: {deductible or "null"}
- prior_payments: {prior_payments or "null"}
- jurisdiction: {jurisdiction or "null"}

DENIAL TEXT (may be empty):
{denial_text}

POLICY EXCERPTS (verbatim snippets, may be empty):
{policy_excerpts}

ESTIMATE TEXT (may be empty):
{estimate_text}

SPOL TEXT (may be empty):
{spol_text}

CLAIM NOTES / CORRESPONDENCE (may be empty):
{notes_text}

RECEIPTS (some may be JSON strings from vision extraction):
{receipts_text}

DAMAGE PHOTO NOTES (may be empty):
{photos_text}
"""

    resp = client.responses.create(
        model=BEST_MODEL,
        input=[{"role": "user", "content": packet_prompt}],
    )
    raw = (resp.output_text or "").strip()

    packet = safe_json_load(raw)
    if not packet:
        # fallback packet
        packet = {
            "insured_name": insured_name or None,
            "insured_property_address": None,
            "carrier_name": None,
            "adjuster_name": None,
            "adjuster_email": None,
            "claim_number": claim_number or None,
            "policy_number": policy_number or None,
            "date_of_loss": None,
            "loss_description": None,
            "timeline": [],
            "denial_reasons_verbatim": [],
            "policy_excerpts_verbatim": [],
            "damages": {"highest_estimate_rcv": None, "highest_estimate_acv": None, "depreciation": None, "notes": "Packet JSON parse failed."},
            "receipts": {"items": [], "total_visible": None},
            "missing_inputs": ["Packet generation failed to return valid JSON."],
            "top_leverage_points": [],
            "_raw_model_output": raw[:8000],
        }

    # attach firm math inputs
    packet["_firm_inputs"] = {
        "deductible": float(deductible),
        "prior_payments": float(prior_payments),
        "jurisdiction": jurisdiction,
        "tone": tone,
    }
    return packet


# =========================
# DRAFT + CHAT REFINEMENT
# =========================
def draft_from_packet(packet: Dict[str, Any]) -> str:
    require_client()

    packet_json = json.dumps(packet, ensure_ascii=False)

    draft_prompt = f"""
You are a top senior first-party property insurance coverage attorney. Literally the best to ever do it.
Use ONLY the Case Fact Packet below. Do not invent facts, policy language, statutes, or case law.

Tone: {tone}

You are a senior first-party property insurance attorney drafting a final pre-litigation demand.

Your writing style must mirror the following characteristics:

• Reads as a cohesive narrative, not a segmented report.
• Transitions naturally between factual background, coverage analysis, and damages.
• Avoids bullet-point lists unless absolutely necessary.
• Escalates tone gradually from factual to firm to litigation-ready.
• Embeds statutory and policy authority seamlessly within paragraphs.
• Applies pressure through chronology and delay where supported by facts.
• Avoids robotic section headings.
• Avoids checklist-style structure.

The letter must:

1. Open by grounding the reader in who the insured is, the property, and the covered loss.
2. Establish the timeline of reporting, inspection, and payment.
3. Contrast the carrier’s payment with the true scope required to restore the property.
4. Explain technical scope differences clearly and persuasively (code compliance, matching, O&P, protective measures, trade coordination, etc. when supported by facts).
5. Tie each denial or underpayment directly to policy language and factual reality.
6. Build legal leverage gradually (statutes, bad faith exposure, delay obligations).
7. Present damages transparently and confidently.
8. End with a firm, controlled litigation-ready demand and deadline.
9. You should have a table that shows the breakdown of the demand. It should have the cost to complete repair the property, a line for "Attorneys' Fees, Bad Faith Release, and Interest" which should be 40% of the total damages rounded to the nearest ten-thousand, and then less prior payments and less deductible. 
10. Write like a human. It should feel like an experienced attorney wrote this. Refrain from em-dashes. 

The letter should feel like it could be attached as Exhibit A to a bad faith complaint.

Do not use bold text.
Do not use unnecessary headings.
Keep under 1800 words.

Find the relevant Unfair Settlement Claims Practice Act for the jurisdiction and cite the statutes that the Company is breaking.
Do not repeat information. It should be succinct while maintaining persuasiveness.

End with the following paragraphs but replace bracketed items with the correct values from the fact sheet: Please send $[demand amount] in check payable to Denham Property and Injury Law Firm F.B.O. [client and PA if applicable] to 250 W. Main St., Suite #120, Lexington, Kentucky 40507 within 14 days. If payment is not received within 14 days we will immediately proceed with litigation seeking all available damages. Failure to do so within the time period provided will be used as further evidence of bad faith. 

Lastly, if [Insurance Company] refuses to make payment within the time period provided, please provide a reasonable explanation of the basis in the Policy in relation to facts and applicable law for its denial to do so. If [Insurance Company] needs additional time to investigate this claim, please provide a reasonable explanation of basis for such need. If you have any questions or would like to discuss, please feel free to reach out.

CASE FACT PACKET (JSON):
{packet_json}
"""

    resp = client.responses.create(
        model=BEST_MODEL,
        input=[{"role": "user", "content": draft_prompt}],
    )
    return (resp.output_text or "").strip()


def refine_draft_chat(packet: Dict[str, Any], current_draft: str, user_instruction: str) -> str:
    require_client()

    packet_json = json.dumps(packet, ensure_ascii=False)
    messages = [
        {"role": "system", "content": "You are refining a property insurance demand letter. Keep edits consistent with the case packet. Do not invent facts or authority."},
        {"role": "user", "content": f"CASE PACKET (JSON):\n{packet_json}"},
        {"role": "user", "content": f"CURRENT DRAFT:\n{current_draft}"},
        {"role": "user", "content": f"USER INSTRUCTION:\n{user_instruction}"},
        {"role": "user", "content": "Revise the draft accordingly. Output ONLY the revised letter body and address block (no letterhead/signature)."},
    ]

    resp = client.responses.create(
        model=BEST_MODEL,
        input=messages,
    )
    return (resp.output_text or "").strip()


# =========================
# EXPORT WORD ON LETTERHEAD
# =========================
def export_on_letterhead(letter_body: str, template_path: str = "Letterhead.docx") -> bytes:
    doc = Document(template_path)
    today_verbose = datetime.now().strftime("%B %d, %Y")

    # Replace date placeholder
    for p in doc.paragraphs:
        if "<< Date.Verbose >>" in p.text:
            p.text = p.text.replace("<< Date.Verbose >>", today_verbose)

    # Find "Dear" line (or fallback)
    insert_index = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().lower().startswith("dear"):
            insert_index = i + 1
            break

    lines = [ln.rstrip() for ln in (letter_body or "").split("\n")]

    if insert_index is not None:
        for line in reversed(lines):
            doc.paragraphs[insert_index].insert_paragraph_before(line)
    else:
        doc.add_paragraph("")
        for line in lines:
            doc.add_paragraph(line)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        tmp.seek(0)
        data = tmp.read()

    try:
        os.remove(tmp.name)
    except Exception:
        pass

    return data


# =========================
# UI
# =========================
st.caption(f"Case ID: {st.session_state.case_id}")

uploaded_files = st.file_uploader(
    "Upload case documents (PDF/DOCX/TXT/CSV + images).",
    accept_multiple_files=True
)

colA, colB, colC = st.columns([1, 1, 1])

with colA:
    if st.button("1) Ingest Documents", use_container_width=True):
        require_client()
        if not uploaded_files:
            st.warning("Upload at least one document.")
            st.stop()

        st.session_state.docs = []
        with st.spinner("Extracting / OCR-ing documents (one-time)…"):
            for f in uploaded_files:
                d = extract_uploaded_file(f)
                # shorten stored raw text? Keep full text for now; packet builder trims as needed.
                st.session_state.docs.append(d)

        st.success(f"Ingested {len(st.session_state.docs)} documents.")

with colB:
    if st.button("2) Build Case Fact Packet", use_container_width=True):
        require_client()
        if not st.session_state.docs:
            st.warning("Ingest documents first.")
            st.stop()

        with st.spinner("Building Case Fact Packet…"):
            st.session_state.case_packet = build_case_packet(st.session_state.docs)

        st.success("Case packet built.")

with colC:
    if st.button("3) Draft Demand (from Packet)", use_container_width=True):
        require_client()
        if not st.session_state.case_packet:
            st.warning("Build the case packet first.")
            st.stop()

        with st.spinner("Drafting demand letter…"):
            st.session_state.draft = draft_from_packet(st.session_state.case_packet)

        # initialize chat history
        st.session_state.chat_messages = []
        st.success("Draft generated.")


st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Case Fact Packet")
    if st.session_state.case_packet:
        st.json(st.session_state.case_packet)
    else:
        st.info("Build the packet to see structured facts, timeline, policy excerpts, damages, receipts, and leverage points.")

with right:
    st.subheader("Demand Draft")
    if st.session_state.draft:
        st.text_area("Current Draft", st.session_state.draft, height=520)

        st.divider()
        st.subheader("Chat Refinement")
        user_edit = st.text_area(
            "Instruction",
            placeholder="e.g., tighten chronology, add elapsed time pressure, make rebuttal sharper, remove speculative language, etc.",
            height=120,
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Apply Refinement", use_container_width=True):
                if not user_edit.strip():
                    st.warning("Type an instruction first.")
                    st.stop()
                with st.spinner("Refining…"):
                    st.session_state.draft = refine_draft_chat(
                        packet=st.session_state.case_packet,
                        current_draft=st.session_state.draft,
                        user_instruction=user_edit.strip(),
                    )
                st.success("Updated draft.")
        with col2:
            if st.button("Reset Draft (re-draft from packet)", use_container_width=True):
                with st.spinner("Re-drafting from packet…"):
                    st.session_state.draft = draft_from_packet(st.session_state.case_packet)
                st.success("Draft reset.")

        st.divider()
        st.subheader("Export on Letterhead")
        try:
            word_bytes = export_on_letterhead(st.session_state.draft, template_path="Letterhead.docx")
            st.download_button(
                label="Download Demand Letter (DOCX)",
                data=word_bytes,
                file_name="Demand_Letter.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Export failed. Ensure Letterhead.docx is in the same folder as app.py. Error: {e}")

    else:
        st.info("Draft a demand from the case packet to enable chat refinement + export.")


  

