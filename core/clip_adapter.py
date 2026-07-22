import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class ClipAdapter:
    """
    Adapter for mapping images and text (or sketches) into a shared 512-D space.
    Uses OpenAI's clip-vit-base-patch32.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None, mock=False):
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

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Takes a PIL Image (sketch or photograph) and returns a normalized 512-D embedding.
        """
        if self.mock:
            # Return a random normalized 512-D vector
            vec = torch.randn(512, device=self.device)
            return vec / vec.norm(p=2, dim=-1, keepdim=True)
            
        # The processor expects RGB images
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        
        # Normalize the embedding
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.squeeze(0) # Returns shape (512,)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Optional utility to map text into the same space if needed for search.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.squeeze(0)
