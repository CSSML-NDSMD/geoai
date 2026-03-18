from flask import Flask, render_template, request, jsonify, session
from google import genai
from google.genai import types
import os, tempfile, base64, time
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=120000)
)

SYSTEM_PROMPT = """You are GeoAI, an intelligent geological analysis assistant.

Your expertise covers:
- Structural geology (faults, folds, joints, lineaments)
- Stratigraphy and sedimentology
- Mineralogy and petrology
- Geomorphology and topographic interpretation
- Remote sensing and geological map reading
- Contour and elevation analysis

Important behavior rules:
- NEVER introduce yourself as a geologist or any human role
- When greeted, respond warmly and say you are here to help with geology-related queries
- Always use standard GSI (Geological Survey of India) and USGS conventions
- Structure your responses with clear headings
- Be technical but precise
- If asked who you are, say: 'I am GeoAI, here to help you with your geology-related queries.'"""

# ── In-memory store for Gemini file references ───────────
# Stores only tiny file reference strings — NOT PDF bytes
file_ref_store = {}


def get_session_key():
    if 'session_key' not in session:
        import uuid
        session['session_key'] = str(uuid.uuid4())
    return session['session_key']


def call_gemini_with_retry(contents, retries=3):
    """Call Gemini with automatic retry on 429/503 errors."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                ),
                contents=contents
            )
            return response
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "503" in err_str) and attempt < retries - 1:
                wait = (attempt + 1) * 30  # 30s then 60s
                time.sleep(wait)
                continue
            raise e


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    tmp_path = None
    try:
        query = request.form.get("query", "").strip()
        if not query:
            return jsonify({"error": "Please enter a geological question."}), 400

        session_key = get_session_key()
        contents = []

        # ── Process Images (always inline) ───────────────
        images = request.files.getlist("images")
        for img_file in images:
            if img_file and img_file.filename:
                img_bytes = img_file.read()

                if len(img_bytes) > 10 * 1024 * 1024:
                    return jsonify({
                        "error": f"Image '{img_file.filename}' exceeds 10MB."
                    }), 400

                fname = img_file.filename.lower()
                if   fname.endswith(".png"):             mime = "image/png"
                elif fname.endswith((".jpg", ".jpeg")):  mime = "image/jpeg"
                elif fname.endswith((".tif", ".tiff")):  mime = "image/tiff"
                else:                                    mime = "image/jpeg"

                contents.append(
                    types.Part.from_bytes(data=img_bytes, mime_type=mime)
                )

        # ── Process PDF via Files API ────────────────────
        pdf_file = request.files.get("pdf")

        if pdf_file and pdf_file.filename:
            # New PDF uploaded — delete old cached ref if exists
            old_ref = file_ref_store.get(session_key)
            if old_ref:
                try:
                    client.files.delete(name=old_ref["file_name"])
                except Exception:
                    pass
                file_ref_store.pop(session_key, None)

            pdf_bytes = pdf_file.read()
            if len(pdf_bytes) > 20 * 1024 * 1024:
                return jsonify({"error": "PDF exceeds 20MB limit."}), 400

            # Write to temp file for upload
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            # Upload to Gemini Files API — lives on Google servers 48hrs
            uploaded = client.files.upload(
                file=tmp_path,
                config=types.UploadFileConfig(
                    mime_type="application/pdf",
                    display_name=pdf_file.filename
                )
            )

            # Store only the tiny file reference string
            file_ref_store[session_key] = {
                "file_name":    uploaded.name,
                "display_name": pdf_file.filename
            }

            contents.append(uploaded)

        elif file_ref_store.get(session_key):
            # Reuse existing cached file reference from Gemini
            ref = file_ref_store[session_key]
            try:
                existing_file = client.files.get(name=ref["file_name"])
                contents.append(existing_file)
            except Exception:
                # File expired — ask user to re-upload
                file_ref_store.pop(session_key, None)
                return jsonify({
                    "error": "Cached PDF expired (48hr limit). Please re-upload your PDF."
                }), 400

        # ── Add question ─────────────────────────────────
        contents.append(query)

        # ── Call Gemini with retry ────────────────────────
        response = call_gemini_with_retry(contents)

        cached_name = file_ref_store.get(session_key, {}).get("display_name")
        return jsonify({
            "result":     response.text,
            "cached_pdf": cached_name
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error: {str(e)}"}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.route("/clear-pdf", methods=["POST"])
def clear_pdf():
    session_key = get_session_key()
    ref = file_ref_store.get(session_key)
    if ref:
        try:
            client.files.delete(name=ref["file_name"])
        except Exception:
            pass
        file_ref_store.pop(session_key, None)
    return jsonify({"status": "cleared"})


@app.route("/generate-image", methods=["POST"])
def generate_image():
    try:
        prompt = request.form.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Please enter an image prompt."}), 400

        full_prompt = (
            f"Professional geological scientific diagram: {prompt}. "
            f"Style: technical, clean, labeled diagram suitable for academic use."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )

        for part in response.parts:
            if part.inline_data is not None:
                img_base64 = base64.b64encode(
                    part.inline_data.data
                ).decode("utf-8")
                mime = part.inline_data.mime_type
                return jsonify({
                    "image": f"data:{mime};base64,{img_base64}"
                })

        return jsonify({
            "error": "No image generated. Try a more descriptive prompt."
        }), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
