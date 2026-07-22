import json
import torch
import random
import os
from typing import List, Dict, Any

try:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

if GENAI_AVAILABLE:
    class GraphConstraints(BaseModel):
        semantic_tags: list[str] = Field(description="A list of relevant tags for searching.")
        style_preference: str = Field(description="A string indicating the likely style (e.g., 'sketch', 'photograph', 'painting', 'any').")
        min_quality: float = Field(description="A float between 0.0 and 1.0 indicating required quality.")

class LLMQueryTranslator:
    """
    Abstraction layer that translates visual features into structured graph database query parameters.
    It performs nearest-neighbor mapping against a standard vocabulary and uses an LLM to generate
    search constraints (tags, metadata).
    """
    def __init__(self, vocab: List[str] = None):
        self.vocab = vocab or ["nature", "city", "portrait", "animal", "abstract", "sketch", "colorful"]
        # Mock embeddings for the vocabulary to simulate nearest-neighbor mapping
        # In a real system, these would be the pre-computed 512-D CLIP embeddings of the vocab
        self.vocab_embeddings = torch.randn(len(self.vocab), 512)
        self.vocab_embeddings = self.vocab_embeddings / self.vocab_embeddings.norm(p=2, dim=-1, keepdim=True)

    def _get_nearest_neighbors(self, visual_vector: torch.Tensor, top_k: int = 3) -> List[str]:
        """
        Map the 512-D vector against the standard vocabulary.
        """
        # Cosine similarity between visual vector and vocab embeddings
        # Ensure they are on the same device
        vocab_embeddings_device = self.vocab_embeddings.to(visual_vector.device)
        similarities = torch.matmul(vocab_embeddings_device, visual_vector.unsqueeze(-1)).squeeze(-1)
        top_indices = torch.topk(similarities, top_k).indices.tolist()
        return [self.vocab[i] for i in top_indices]

    def _generate_prompt(self, visual_concepts: List[str]) -> str:
        """
        Create a lightweight prompt for the LLM.
        """
        concepts_str = ", ".join(visual_concepts)
        prompt = f"""
        You are a visual search assistant. The user has provided an image or sketch that strongly aligns with the following concepts: {concepts_str}.
        Translate these concepts into structured metadata constraints for a graph database search.
        
        Provide your response as a JSON object with the following keys:
        - "semantic_tags": A list of relevant tags for searching.
        - "style_preference": A string indicating the likely style (e.g., "sketch", "photograph", "painting", "any").
        - "min_quality": A float between 0.0 and 1.0 indicating required quality (default to 0.5).
        
        Respond ONLY with valid JSON.
        """
        return prompt.strip()

    def translate_vector(self, visual_vector: torch.Tensor) -> Dict[str, Any]:
        """
        Takes the 512-D output vector, finds nearest neighbors, and uses a mock LLM 
        to return structured search constraints.
        """
        if visual_vector.shape != (512,):
            raise ValueError(f"Expected visual vector of shape (512,), got {visual_vector.shape}")

        nearest_concepts = self._get_nearest_neighbors(visual_vector)
        prompt = self._generate_prompt(nearest_concepts)
        
        print("--- LLM Prompt Generated ---")
        print(prompt)
        print("----------------------------")
        
        if GENAI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
            print("Querying Gemini via google-genai...")
            try:
                client = genai.Client()
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GraphConstraints,
                        temperature=0.1
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                print(f"GenAI Error: {e}. Falling back to mock.")
        else:
            print("google-genai not installed or GEMINI_API_KEY not set. Using mock LLM response.")

        # Mocking the LLM response
        mock_response = {
            "semantic_tags": nearest_concepts + ["custom_tag"],
            "style_preference": "sketch" if "sketch" in nearest_concepts else "any",
            "min_quality": round(random.uniform(0.5, 0.8), 2)
        }
        
        return mock_response
