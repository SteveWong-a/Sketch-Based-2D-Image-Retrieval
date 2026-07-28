import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os

class LLMQueryTranslator:
    """
    Abstraction layer that translates visual features into open-ended search descriptions.
    translate_mode="direct" (default) — joins top-5 CLIP concepts into a clean phrase.
    translate_mode="llm"    — uses local flan-t5-small to generate a sentence (slower).
    """
    def __init__(self, clip_adapter, vocab=None, translate_mode="direct"):
        self.translate_mode = translate_mode
        
        if vocab is None:
            vocab_path = os.path.join(os.path.dirname(__file__), "vocab.json")
            if os.path.exists(vocab_path):
                with open(vocab_path, "r") as f:
                    self.vocab = json.load(f)
            else:
                self.vocab = ["nature", "city", "portrait", "animal", "abstract", "sketch", "colorful"]
            
            if not self.vocab:
                self.vocab = ["nature", "city", "portrait", "animal", "abstract", "sketch", "colorful"]
        else:
            self.vocab = vocab
            
        # --- Vocab Embedding (with disk cache) ---
        cache_path = os.path.join(os.path.dirname(__file__), "..", "data", "vocab_embeddings.pt")
        cache_path = os.path.normpath(cache_path)
        
        if os.path.exists(cache_path):
            print(f"Loading cached vocab embeddings from {cache_path}")
            self.vocab_embeddings = torch.load(cache_path, map_location="cpu")
        else:
            print("Computing vocab embeddings (first run — will cache for next time)...")
            vocab_tensors = []
            for word in self.vocab:
                embed = clip_adapter.encode_text(word)
                vocab_tensors.append(embed.cpu())
            self.vocab_embeddings = torch.stack(vocab_tensors)
            torch.save(self.vocab_embeddings, cache_path)
            print(f"Vocab embeddings cached to {cache_path}")
        
        # --- Lazy-load flan-t5 only when needed ---
        self.tokenizer = None
        self.model = None
        if self.translate_mode == "llm":
            print("Loading local LLM (google/flan-t5-small) for translation...")
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    def _get_nearest_neighbors(self, visual_vector: torch.Tensor, top_k: int = 5) -> list:
        vocab_embeddings_device = self.vocab_embeddings.to(visual_vector.device)
        similarities = torch.matmul(vocab_embeddings_device, visual_vector.unsqueeze(-1)).squeeze(-1)
        top_indices = torch.topk(similarities, top_k).indices.tolist()
        return [self.vocab[i] for i in top_indices]

    def get_confidence(self, visual_vector: torch.Tensor) -> float:
        """Returns top-1 cosine similarity score as a recognition confidence (0.0 – 1.0)."""
        vocab_embeddings_device = self.vocab_embeddings.to(visual_vector.device)
        similarities = torch.matmul(vocab_embeddings_device, visual_vector.unsqueeze(-1)).squeeze(-1)
        top_score = similarities.max().item()
        # Cosine sim in CLIP space is typically 0.15–0.35 for good matches
        # Normalise to a 0-1 range using empirical bounds
        normalised = max(0.0, min(1.0, (top_score - 0.15) / 0.20))
        return round(normalised, 3)

    def _generate_prompt(self, visual_concepts: list) -> str:
        concepts_str = ", ".join(visual_concepts)
        return f"Describe an image that features the following concepts: {concepts_str}. Give a short description for a search query."

    def translate_vector(self, visual_vector: torch.Tensor) -> str:
        if visual_vector.shape != (512,):
            raise ValueError(f"Expected visual vector of shape (512,), got {visual_vector.shape}")

        nearest_concepts = self._get_nearest_neighbors(visual_vector)
        
        if self.translate_mode == "direct":
            # Join the top concepts into a clean search phrase — fast and accurate
            translation = " ".join(nearest_concepts)
            print(f"--- Direct Concept Translation: {translation} ---")
            return translation
        else:
            # flan-t5 LLM path (legacy, slower, prone to hallucination)
            prompt = self._generate_prompt(nearest_concepts)
            print("--- Local LLM Prompt Generated ---")
            print(prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=50)
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated Description: {translation}")
            print("----------------------------------")
            return translation
