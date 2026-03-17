from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # Reduce to 20MB for free tier

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=120000)
)

SYSTEM_PROMPT = """You are GeoAI, an intelligent geological analysis assistant:
- Structural geology (faults, folds, joints, lineaments)
- Stratigraphy and sedimentology
- Mineralogy and petrology
- Geomorphology and topographic interpretation
- Remote sensing and geological map reading
- Contour and elevation analysis

Always use standard GSI (Geological Survey of India) and USGS conventions.
Structure your responses with clear headings. Be technical but precise.

Important behavior rules:
- NEVER introduce yourself as a geologist or any human role
- When greeted, respond warmly and simply say you are here to help with geology-related queries
- Always use standard GSI (Geological Survey of India) and USGS conventions
- Structure your responses with clear headings
- Be technical but precise
- If asked who you are, say: 'I am GeoAI, here to help you with your geology-related queries.'"""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        query = request.form.get("query", "").strip()
        if not query:
            return jsonify({"error": "Please enter a geological question."}), 400

        contents = []

        # ── Process Images ───────────────────────────────
        images = request.files.getlist("images")
        for img_file in images:
            if img_file and img_file.filename:
                img_bytes = img_file.read()

                if len(img_bytes) > 10 * 1024 * 1024:
                    return jsonify({
                        "error": f"Image '{img_file.filename}' exceeds 10MB. Please resize it."
                    }), 400

                fname = img_file.filename.lower()
                if   fname.endswith(".png"):             mime = "image/png"
                elif fname.endswith((".jpg", ".jpeg")):  mime = "image/jpeg"
                elif fname.endswith((".tif", ".tiff")):  mime = "image/tiff"
                else:                                    mime = "image/jpeg"

                contents.append(
                    types.Part.from_bytes(data=img_bytes, mime_type=mime)
                )

        # ── Process PDF directly (no temp file) ─────────
        pdf_file = request.files.get("pdf")
        if pdf_file and pdf_file.filename:
            pdf_bytes = pdf_file.read()

            if len(pdf_bytes) > 20 * 1024 * 1024:
                return jsonify({
                    "error": "PDF exceeds 20MB limit. Please use a smaller file."
                }), 400

            # Send directly as bytes — no temp file, no Files API
            contents.append(
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf"
                )
            )

        # ── Add question ─────────────────────────────────
        contents.append(query)

        # ── Call Gemini ──────────────────────────────────
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


if __name__ == "__main__":
    app.run(debug=True)
