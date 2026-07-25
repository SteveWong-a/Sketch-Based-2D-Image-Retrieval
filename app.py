import os
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
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

@app.route("/api/process", methods=["POST"])
def process_sketch():
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        image_data = data["image"].split(',')[1] # Remove data:image/png;base64,
        image_bytes = base64.b64decode(image_data)
        sketch = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # 1. Extract CLIP Embedding
        clip_embedding = clip_adapter.encode_image(sketch)
        
        # 2. Translate to Graph Constraints via Local LLM
        translation_desc = llm_translator.translate_vector(clip_embedding)
        
        # 3. Generate Cypher Query (Open Vector Search)
        cypher_query = graph_db.early_fusion_search(
            query_vector=clip_embedding.tolist(),
            metadata_constraints=None,
            top_k=5
        )
        
        # Extract concepts so we can pass to the mock db to get better images
        concepts = llm_translator._get_nearest_neighbors(clip_embedding)
        
        # 4. Execute Query (Mock) to get Image Results
        # Passing the concepts so the mock API returns visually relevant images
        image_results = graph_db.mock_execute_query(cypher_query, concepts=concepts)
        
        return jsonify({
            "success": True,
            "embedding_shape": list(clip_embedding.shape),
            "translation": translation_desc,
            "cypher_query": cypher_query,
            "results": image_results,
            "concepts_prompt": llm_translator._generate_prompt(concepts)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the server
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
