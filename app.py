import os
import json
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from core.clip_adapter import ClipAdapter
from core.llm_translator import LLMQueryTranslator
from graph_db.connector import GraphDBConnector
import torch

app = Flask(__name__, static_folder="static")
CORS(app)

print("Initializing Pipeline Components...")
clip_adapter = ClipAdapter()
llm_translator = LLMQueryTranslator(clip_adapter=clip_adapter)
graph_db = GraphDBConnector()
print("Components Ready!")

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# ─────────────────────────────────────────────────────────────
# Standard (non-streaming) endpoint — kept for compatibility
# ─────────────────────────────────────────────────────────────
@app.route("/api/process", methods=["POST"])
def process_sketch():
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data["image"].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        sketch = Image.open(BytesIO(image_bytes)).convert("RGB")

        clip_embedding = clip_adapter.encode_image(sketch)
        concepts = llm_translator._get_nearest_neighbors(clip_embedding)
        confidence = llm_translator.get_confidence(clip_embedding)
        translation_desc = llm_translator.translate_vector(clip_embedding)

        cypher_query = graph_db.early_fusion_search(
            query_vector=clip_embedding.tolist(),
            metadata_constraints=None,
            top_k=5
        )

        image_results = graph_db.mock_execute_query(cypher_query, concepts=concepts)

        return jsonify({
            "success": True,
            "embedding_shape": list(clip_embedding.shape),
            "concepts": concepts,
            "translation": translation_desc,
            "confidence": confidence,
            "cypher_query": cypher_query,
            "results": image_results,
            "concepts_prompt": llm_translator._generate_prompt(concepts)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# SSE streaming endpoint — step-by-step pipeline progress
# ─────────────────────────────────────────────────────────────
@app.route("/api/process/stream", methods=["POST"])
def process_sketch_stream():
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data["image"].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        sketch = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    def generate():
        def sse(event, payload):
            return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

        try:
            # Step 1: CLIP Embedding
            clip_embedding = clip_adapter.encode_image(sketch)
            yield sse("clip_done", {
                "embedding_shape": list(clip_embedding.shape)
            })

            # Step 2: Confidence + Concepts
            confidence = llm_translator.get_confidence(clip_embedding)
            concepts = llm_translator._get_nearest_neighbors(clip_embedding)
            yield sse("concepts_done", {
                "concepts": concepts,
                "confidence": confidence
            })

            # Step 3: Translation + Cypher
            translation_desc = llm_translator.translate_vector(clip_embedding)
            cypher_query = graph_db.early_fusion_search(
                query_vector=clip_embedding.tolist(),
                metadata_constraints=None,
                top_k=5
            )
            yield sse("cypher_done", {
                "translation": translation_desc,
                "cypher_query": cypher_query
            })

            # Step 4: Image Results
            image_results = graph_db.mock_execute_query(cypher_query, concepts=concepts)
            yield sse("results_done", {
                "results": image_results
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield sse("error", {"message": str(e)})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
