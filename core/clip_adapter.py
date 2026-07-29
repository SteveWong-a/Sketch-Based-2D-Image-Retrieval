import torch
from PIL import Image, ImageOps, ImageEnhance
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

class ClipAdapter:
    """
    Adapter for mapping images and text (or sketches) into a shared space.
    Uses TinyCLIP (wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M) for extreme memory efficiency.
    """
    def __init__(self, model_name="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", device=None, mock=False):
        self.mock = mock
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        else:
            self.device = device
            
        if self.mock:
            print(f"Loading MOCK CLIP model on {self.device} for testing...")
            return
            
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def preprocess_sketch(self, image: Image.Image) -> Image.Image:
        """
        Improves CLIP embedding quality for hand-drawn sketches:
        - Inverts (black bg, white lines) to match edge-map training distribution
        - Boosts contrast to sharpen faint strokes
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Detect if this looks like a sketch (predominantly white bg)
        import numpy as np
        arr = np.array(image)
        mean_brightness = arr.mean()
        
        if mean_brightness > 200:  # Light background — treat as sketch
            image = ImageOps.invert(image)
            image = ImageEnhance.Contrast(image).enhance(1.5)
        
        return image

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Takes a PIL Image (sketch or photograph) and returns a normalized 512-D embedding.
        """
        if self.mock:
            # Return a random normalized 512-D vector
            vec = torch.randn(512, device=self.device)
            return F.normalize(vec, p=2, dim=-1)
            
        # Pre-process sketch before encoding
        image = self.preprocess_sketch(image)
            
        # The processor expects RGB images
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features_out = self.model.get_image_features(**inputs)
        
        if hasattr(image_features_out, "pooler_output"):
            image_features = image_features_out.pooler_output
        elif hasattr(image_features_out, "image_embeds"):
            image_features = image_features_out.image_embeds
        else:
            image_features = image_features_out
        
        # Normalize the embedding
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features.squeeze(0) # Returns shape (512,)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Optional utility to map text into the same space if needed for search.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        text_features_out = self.model.get_text_features(**inputs)
        
        if hasattr(text_features_out, "pooler_output"):
            text_features = text_features_out.pooler_output
        elif hasattr(text_features_out, "text_embeds"):
            text_features = text_features_out.text_embeds
        else:
            text_features = text_features_out
            
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features.squeeze(0)
