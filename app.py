from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
import os, tempfile
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=120000)  # 120 seconds
)

SYSTEM_PROMPT = """You are a senior exploration geologist with deep expertise in:
- Structural geology (faults, folds, joints, lineaments)
- Stratigraphy and sedimentology
- Mineralogy and petrology
- Geomorphology and topographic interpretation
- Remote sensing and geological map reading
- Contour and elevation analysis

Always use standard GSI (Geological Survey of India) and USGS conventions.
Structure your responses with clear headings. Be technical but precise."""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded_refs = []  # Track all uploaded file refs for cleanup
    tmp_path = None

    try:
        query = request.form.get("query", "").strip()
        if not query:
            return jsonify({"error": "Please enter a geological question."}), 400

        contents = []

        # ── Process Images (inline) ──────────────────────
        images = request.files.getlist("images")
        for img_file in images:
            if img_file and img_file.filename:
                img_bytes = img_file.read()

                if len(img_bytes) > 100 * 1024 * 1024:
                    return jsonify({
                        "error": f"Image '{img_file.filename}' exceeds 100MB limit."
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
            pdf_bytes = pdf_file.read()

            if len(pdf_bytes) > 100 * 1024 * 1024:
                return jsonify({
                    "error": f"PDF '{pdf_file.filename}' exceeds 100MB limit."
                }), 400

            # Write to temp file for upload
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            ) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            # Upload to Gemini Files API
            file_ref = client.files.upload(
                file=tmp_path,
                config=types.UploadFileConfig(
                    mime_type="application/pdf",
                    display_name=pdf_file.filename
                )
            )
            uploaded_refs.append(file_ref)
            contents.append(file_ref)

        # ── Add question ─────────────────────────────────
        contents.append(query)

        # ── Call Gemini 2.5 Pro ──────────────────────────
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            ),
            contents=contents
        )

        return jsonify({"result": response.text})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error: {str(e)}"}), 500

    finally:
        # Clean up temp file from disk
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        # Delete uploaded files from Gemini after response
        for ref in uploaded_refs:
            try:
                client.files.delete(name=ref.name)
            except Exception:
                pass


if __name__ == "__main__":
    app.run(debug=True)
