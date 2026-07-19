import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Import arsitektur hybrid lokal
from hybrid_model import HybridSequentialModelOptB

def _load_checkpoint(model_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    return checkpoint, device

def _build_hybrid_model(num_classes: int, state_dict: dict, device):
    config = {
        'proj_dropout': 0.0,
        'pos_dropout': 0.0,
        'mamba_bidirectional': True, 
        'mamba_drop_path': 0.0,
        'learning_rate': 1e-4, 
        'weight_decay': 1e-4   
    }
    model = HybridSequentialModelOptB(num_classes=num_classes, config=config)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def _parse_norm(mean, std):
    return [float(v) for v in mean], [float(v) for v in std]

def load_model_gatekeeper_hybrid(model_path: str):
    checkpoint, device = _load_checkpoint(model_path)
    class_names = checkpoint.get('class_names', ['Bukan Kulit', 'Kulit'])
    mean, std   = _parse_norm(checkpoint['mean'], checkpoint['std'])
    model = _build_hybrid_model(len(class_names), checkpoint['model_state_dict'], device)
    return model, class_names, mean, std

def load_model_disease_hybrid(model_path: str):
    checkpoint, device = _load_checkpoint(model_path)
    class_names = checkpoint['class_names']
    mean, std   = _parse_norm(checkpoint['mean'], checkpoint['std'])
    model = _build_hybrid_model(len(class_names), checkpoint['model_state_dict'], device)
    return model, class_names, mean, std

def _preprocess(image: Image.Image, mean, std) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform(image.convert('RGB')).unsqueeze(0)

@torch.no_grad()
def predict_gatekeeper_hybrid(model, image: Image.Image, mean, std):
    device = next(model.parameters()).device
    tensor = _preprocess(image, mean, std).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze(0)
    return probs.cpu().numpy(), int(probs.argmax().item())

@torch.no_grad()
def predict_disease_hybrid(model, image: Image.Image, mean, std):
    device = next(model.parameters()).device
    tensor = _preprocess(image, mean, std).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze(0)
    return probs.cpu().numpy(), int(probs.argmax().item())