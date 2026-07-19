import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Import cetak biru arsitektur Mamba
from models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

def _load_checkpoint(model_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    return checkpoint, device

def _build_mamba_model(num_classes: int, state_dict: dict, device):
    # 1. Bangun KERANGKA KOSONG (pretrained=False, tidak mengunduh apapun)
    model = vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
        pretrained=False, 
        num_classes=1000, 
        if_bidirectional=True
    )
    
    # 2. Sesuaikan ujung model (head) agar pas dengan jumlah kelas Anda
    in_features = model.head.in_features
    model.head  = nn.Linear(in_features, num_classes)

    # 3. Bersihkan nama layer jika sebelumnya dilatih menggunakan DataParallel (multi-GPU)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    # 4. SUNTIKKAN BOBOT ANDA (.pth) ke dalam kerangka
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def _parse_norm(mean, std):
    return [float(v) for v in mean], [float(v) for v in std]

def load_model_gatekeeper_mamba(model_path: str):
    checkpoint, device = _load_checkpoint(model_path)
    class_names = checkpoint.get('class_names', ['Bukan Kulit', 'Kulit'])
    mean, std   = _parse_norm(checkpoint['mean'], checkpoint['std'])
    model = _build_mamba_model(len(class_names), checkpoint['model_state_dict'], device)
    return model, class_names, mean, std

def load_model_disease_mamba(model_path: str):
    checkpoint, device = _load_checkpoint(model_path)
    class_names = checkpoint['class_names']
    mean, std   = _parse_norm(checkpoint['mean'], checkpoint['std'])
    model = _build_mamba_model(len(class_names), checkpoint['model_state_dict'], device)
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
def predict_gatekeeper_mamba(model, image: Image.Image, mean, std):
    device = next(model.parameters()).device
    tensor = _preprocess(image, mean, std).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze(0)
    return probs.cpu().numpy(), int(probs.argmax().item())

@torch.no_grad()
def predict_disease_mamba(model, image: Image.Image, mean, std):
    device = next(model.parameters()).device
    tensor = _preprocess(image, mean, std).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze(0)
    return probs.cpu().numpy(), int(probs.argmax().item())