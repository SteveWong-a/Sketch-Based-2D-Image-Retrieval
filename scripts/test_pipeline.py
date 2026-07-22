import torch
from PIL import Image
import numpy as np

from core.clip_adapter import ClipAdapter
from core.llm_translator import LLMQueryTranslator
from graph_db.connector import GraphDBConnector
from graph_db.schema import Neo4jSchemaManager

def test_pipeline():
    print("=== Testing Sketch-Based 2D Image Retrieval Pipeline ===\n")
    
    # 1. Initialize Components
    print("1. Initializing components...")
    clip_adapter = ClipAdapter()
    llm_translator = LLMQueryTranslator()
    graph_db = GraphDBConnector()
    print("Components initialized successfully.\n")
    
    # 2. Simulate an input sketch (using a random white/gray image)
    print("2. Simulating input sketch...")
    # Create a dummy 224x224 RGB image
    dummy_sketch = Image.fromarray(np.random.randint(200, 255, (224, 224, 3), dtype=np.uint8))
    
    # 3. Extract 512-D CLIP Embedding
    print("3. Extracting CLIP Embedding...")
    clip_embedding = clip_adapter.encode_image(dummy_sketch)
    print(f"Extracted embedding shape: {clip_embedding.shape}")
    
    # 4. LLM Translation Layer
    print("\n4. Translating features to graph constraints via mock LLM...")
    constraints = llm_translator.translate_vector(clip_embedding)
    print(f"Structured Constraints: {constraints}\n")
    
    # 5. Early-Fusion Graph Query
    print("5. Generating Early-Fusion Cypher Query...")
    cypher_query = graph_db.early_fusion_search(
        query_vector=clip_embedding.tolist(),
        metadata_constraints=constraints,
        top_k=5
    )
    print("Generated Query:")
    print(cypher_query)
    
    print("\n=== Pipeline Test Complete ===")

if __name__ == "__main__":
    test_pipeline()
