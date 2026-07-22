import json
import torch
import random
from typing import List, Dict, Any

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
        similarities = torch.matmul(self.vocab_embeddings, visual_vector.unsqueeze(-1)).squeeze(-1)
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
        
        # Mocking the LLM response
        mock_response = {
            "semantic_tags": nearest_concepts + ["custom_tag"],
            "style_preference": "sketch" if "sketch" in nearest_concepts else "any",
            "min_quality": round(random.uniform(0.5, 0.8), 2)
        }
        
        return mock_response
