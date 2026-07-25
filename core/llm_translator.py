import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os

class LLMQueryTranslator:
    """
    Abstraction layer that translates visual features into open-ended search descriptions
    using a local Hugging Face LLM (no API keys required).
    """
    def __init__(self, clip_adapter, vocab=None):
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
            
        print("Embedding vocabulary with CLIP...")
        vocab_tensors = []
        for word in self.vocab:
            embed = clip_adapter.encode_text(word)
            vocab_tensors.append(embed)
        self.vocab_embeddings = torch.stack(vocab_tensors)
        
        # Load local text-to-text model directly
        print("Loading local LLM (google/flan-t5-small) for translation...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    def _get_nearest_neighbors(self, visual_vector: torch.Tensor, top_k: int = 3) -> list:
        vocab_embeddings_device = self.vocab_embeddings.to(visual_vector.device)
        similarities = torch.matmul(vocab_embeddings_device, visual_vector.unsqueeze(-1)).squeeze(-1)
        top_indices = torch.topk(similarities, top_k).indices.tolist()
        return [self.vocab[i] for i in top_indices]

    def _generate_prompt(self, visual_concepts: list) -> str:
        concepts_str = ", ".join(visual_concepts)
        return f"Describe an image that features the following concepts: {concepts_str}. Give a short description for a search query."

    def translate_vector(self, visual_vector: torch.Tensor) -> str:
        if visual_vector.shape != (512,):
            raise ValueError(f"Expected visual vector of shape (512,), got {visual_vector.shape}")

        nearest_concepts = self._get_nearest_neighbors(visual_vector)
        prompt = self._generate_prompt(nearest_concepts)
        
        print("--- Local LLM Prompt Generated ---")
        print(prompt)
        
        # Run local generation
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Generated Description: {translation}")
        print("----------------------------------")
        
        return translation
