import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image

# =============================================================================
# KONFIGURASI KELAS (Bisa diimpor oleh app.py)
# =============================================================================
CLASS_DESCRIPTIONS = {
    'akiec': 'Actinic Keratosis / Intraepithelial Carcinoma',
    'bcc'  : 'Basal Cell Carcinoma (Karsinoma Sel Basal)',
    'bkl'  : 'Benign Keratosis-like Lesion (Lesi Jinak Keratosis)',
    'df'   : 'Dermatofibroma',
    'mel'  : 'Melanoma (Kanker Kulit Ganas)',
    'nv'   : 'Melanocytic Nevi (Tahi Lalat / Nevus)',
    'vasc' : 'Vascular Lesion (Lesi Vaskular)',
}

DISEASE_DESCRIPTION = {
    'akiec': 'Lesi pra-kanker yang kasar dan bersisik pada permukaan kulit akibat paparan sinar matahari kronis.',
    'bcc'  : 'Jenis kanker kulit paling umum yang tumbuh lambat, jarang menyebar, dan sering muncul di area yang terpapar matahari.',
    'bkl'  : 'Kelompok lesi kulit jinak non-kanker yang memiliki tampilan menyerupai kutil atau bercak kerak kecokelatan.',
    'df'   : 'Benjolan kecil bersifat jinak pada jaringan ikat kulit yang biasanya terasa keras dan berwarna gelap.',
    'mel'  : 'Kanker kulit paling agresif dan berbahaya yang berkembang dari sel pigmen (melanosit) dengan risiko penyebaran tinggi.',
    'nv'   : 'Pertumbuhan pigmen kulit yang sangat umum dan bersifat jinak, secara medis dikenal sebagai tahi lalat.',
    'vasc' : 'Kelainan atau pertumbuhan pada pembuluh darah di kulit yang biasanya tampak sebagai bercak merah atau keunguan.'
}

CLASS_RISK = {'akiec': 'Tinggi', 'bcc': 'Tinggi', 'mel': 'Tinggi', 'vasc': 'Sedang', 'bkl': 'Rendah', 'df': 'Rendah', 'nv': 'Rendah'}
CLASS_RISK_COLOR = {'Tinggi': '#E53E3E', 'Sedang': '#DD6B20', 'Rendah': '#38A169'}
CLASS_RISK_ICON = {'Tinggi': '🔴', 'Sedang': '🟡', 'Rendah': '🟢'}

def _load_checkpoint(model_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    return checkpoint, device

def _build_model(model_name: str, num_classes: int, state_dict: dict, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def _parse_norm(mean, std):
    return [float(v) for v in mean], [float(v) for v in std]

def load_model_gatekeeper_effnet(model_path: str):
    checkpoint, device = _load_checkpoint(model_path)
    class_names = checkpoint.get('class_names', ['Bukan Kulit', 'Kulit'])
    mean, std   = _parse_norm(checkpoint['mean'], checkpoint['std'])
    model = _build_model(checkpoint.get('model_architecture', 'efficientnet_b0'), len(class_names), checkpoint['model_state_dict'], device)
    return model, class_names, mean, std

def load_model_disease_effnet(model_path: str):
    checkpoint, device = _load_checkpoint(model_path)
    class_names = checkpoint['class_names']
    mean, std   = _parse_norm(checkpoint['mean'], checkpoint['std'])
    model = _build_model(checkpoint.get('model_architecture', 'efficientnet_b0'), len(class_names), checkpoint['model_state_dict'], device)
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
def predict_gatekeeper_effnet(model, image: Image.Image, mean, std):
    device = next(model.parameters()).device
    tensor = _preprocess(image, mean, std).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze(0)
    return probs.cpu().numpy(), int(probs.argmax().item())

@torch.no_grad()
def predict_disease_effnet(model, image: Image.Image, mean, std):
    device = next(model.parameters()).device
    tensor = _preprocess(image, mean, std).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze(0)
    return probs.cpu().numpy(), int(probs.argmax().item())