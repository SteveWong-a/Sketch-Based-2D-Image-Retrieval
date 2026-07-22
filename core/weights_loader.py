import torch
import os
import logging

logger = logging.getLogger(__name__)

def load_baseline_model(model_path: str, device=None) -> dict:
    """
    Secure utility loader to ingest a pre-trained model checkpoint.
    Uses weights_only=True to prevent arbitrary code execution from malicious pickles.
    
    Args:
        model_path: Path to the .pth or .pt file
        device: Torch device to load the weights onto
        
    Returns:
        The state dictionary of the loaded model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Baseline model not found at {model_path}")
        
    logger.info(f"Loading baseline model from {model_path} onto {device} securely...")
    
    try:
        # weights_only=True ensures we only load tensors, not arbitrary Python objects
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        logger.info("Successfully loaded baseline model weights.")
        return state_dict
    except Exception as e:
        logger.error(f"Failed to load baseline model: {e}")
        raise

class BaselineFeatureExtractor(torch.nn.Module):
    """
    Wrapper class to anchor the pre-trained model checkpoint (skribbl_model.pth) 
    as the baseline feature extractor for progressive drawings.
    Maps the 300-D output to the 512-D CLIP embedding space.
    """
    def __init__(self, model_class, model_path: str, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        self.backbone = model_class().to(self.device)
        
        state_dict = load_baseline_model(model_path, self.device)
        self.backbone.load_state_dict(state_dict)
        self.backbone.eval()
        
        # Projection layer to map 300-D to 512-D CLIP space
        self.projection = torch.nn.Linear(300, 512).to(self.device)
        
    @torch.no_grad()
    def extract_features(self, x, lengths) -> torch.Tensor:
        """
        Extract 512-D features from the progressive drawing.
        """
        backbone_features = self.backbone(x, lengths)
        clip_space_features = self.projection(backbone_features)
        # Normalize to match CLIP embedding norm
        clip_space_features = clip_space_features / clip_space_features.norm(p=2, dim=-1, keepdim=True)
        return clip_space_features
