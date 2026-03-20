from flask import Flask, render_template, request, jsonify, session
from google import genai
from google.genai import types
import os, tempfile, base64, time
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

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
- Do NOT start your answers with greetings or introductions like "Hello! I am GeoAI...". Just answer directly.
- Always use standard GSI (Geological Survey of India) and USGS conventions
- Structure your responses with clear headings
- Be technical but precise
- CRITICAL: If the user uploads both an image and a document, read BOTH carefully and synthesize the final answer. Do NOT ignore the image.
- If explicitly asked who you are, say: 'I am GeoAI, here to help you with your geology-related queries.'"""

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

                if len(img_bytes) > 20 * 1024 * 1024:
                    return jsonify({
                        "error": f"Image '{img_file.filename}' exceeds 20MB."
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
            if old_ref and old_ref.get("file_name"):
                try:
                    client.files.delete(name=old_ref["file_name"])
                except Exception:
                    pass
            file_ref_store.pop(session_key, None)

            pdf_bytes = pdf_file.read()
            if len(pdf_bytes) > 20 * 1024 * 1024:
                return jsonify({"error": "Document exceeds 20MB limit."}), 400

            fname = pdf_file.filename.lower()

            if fname.endswith(".docx"):
                import io, docx
                doc = docx.Document(io.BytesIO(pdf_bytes))
                text = "\n".join([p.text for p in doc.paragraphs])
                file_ref_store[session_key] = {
                    "type": "text",
                    "content": text,
                    "display_name": pdf_file.filename
                }
                contents.append(f"Document Content from {pdf_file.filename}:\n\n{text}")

            elif fname.endswith(".txt"):
                text = pdf_bytes.decode("utf-8", errors="replace")
                file_ref_store[session_key] = {
                    "type": "text",
                    "content": text,
                    "display_name": pdf_file.filename
                }
                contents.append(f"Document Content from {pdf_file.filename}:\n\n{text}")

            else:
                # Write to temp file for upload (assuming PDF)
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
                    "type": "pdf",
                    "file_name":    uploaded.name,
                    "display_name": pdf_file.filename
                }

                contents.append(uploaded)

        elif request.form.get("use_cache") == "true" and file_ref_store.get(session_key):
            ref = file_ref_store[session_key]
            if ref.get("type") == "text":
                contents.append(f"Document Content from {ref['display_name']}:\n\n{ref['content']}")
            else:
                try:
                    existing_file = client.files.get(name=ref["file_name"])
                    contents.append(existing_file)
                except Exception:
                    # File expired — ask user to re-upload
                    file_ref_store.pop(session_key, None)
                    return jsonify({
                        "error": "Cached document expired (48hr limit). Please re-upload your document."
                    }), 400

        # ── Add question ─────────────────────────────────
        has_doc = (pdf_file and pdf_file.filename) or (request.form.get("use_cache") == "true" and file_ref_store.get(session_key))
        has_img = len(images) > 0
        
        if has_doc and has_img:
            contents.append("\n[SYSTEM NOTE: The user has provided BOTH image(s) and a document. You MUST analyze and address BOTH sources to answer the query correctly.]\n")

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
        if ref.get("type") != "text":
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

        response = client.models.generate_images(
            model="imagen-4.0-generate-001",
            prompt=full_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="1:1"
            )
        )

        for generated_image in response.generated_images:
            img_bytes = generated_image.image.image_bytes
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            return jsonify({
                "image": f"data:image/jpeg;base64,{img_base64}"
            })

        return jsonify({
            "error": "No image generated. Try a more descriptive prompt."
        }), 400

    except Exception as e:
        err_str = str(e)
        if "paid plans" in err_str.lower() or "INVALID_ARGUMENT" in err_str:
             return jsonify({"error": "Image generation (Imagen 3/4) requires a paid Google AI Studio plan. It is not available on the free tier."}), 400
        
        import traceback
        traceback.print_exc()
        return jsonify({"error": err_str}), 500


if __name__ == "__main__":
    app.run(debug=True)
