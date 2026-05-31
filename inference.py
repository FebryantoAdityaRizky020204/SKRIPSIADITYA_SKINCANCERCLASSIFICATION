import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image
from matplotlib import cm as mpl_cm

# =============================================================================
# KONFIGURASI KELAS
# =============================================================================
# CLASS_NAMES diambil langsung dari checkpoint saat load_model()

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

CLASS_RISK = {
    'akiec': 'Tinggi',
    'bcc'  : 'Tinggi',
    'mel'  : 'Tinggi',
    'vasc' : 'Sedang',
    'bkl'  : 'Rendah',
    'df'   : 'Rendah',
    'nv'   : 'Rendah',
}

CLASS_RISK_COLOR = {
    'Tinggi': '#E53E3E',
    'Sedang': '#DD6B20',
    'Rendah': '#38A169',
}

CLASS_RISK_ICON = {
    'Tinggi': '🔴',
    'Sedang': '🟡',
    'Rendah': '🟢',
}

# =============================================================================
# LOAD MODEL — membaca mean, std, dan class_names langsung dari checkpoint
# =============================================================================
def load_model(model_path: str):
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # weights_only=False diperlukan karena checkpoint menyimpan numpy array
    # (mean, std, class_names) yang diblokir oleh default PyTorch 2.6
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    class_names = checkpoint['class_names']
    mean        = checkpoint['mean']
    std         = checkpoint['std']

    # Pastikan mean dan std dalam format list of float
    mean = [float(v) for v in mean]
    std  = [float(v) for v in std]

    model = timm.create_model(
        'efficientnet_b0',
        pretrained  = False,
        num_classes = len(class_names),
    )

    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded  : {model_path}")
    print(f"Device        : {device}")
    print(f"Classes       : {class_names}")
    print(f"Norm mean     : {mean}")
    print(f"Norm std      : {std}")
    return model, class_names, mean, std


# =============================================================================
# INFERENCE
# =============================================================================
@torch.no_grad()
def predict(model, image: Image.Image, mean, std):
    """
    Preprocessing konsisten dengan training:
    - Resize ke 224×224
    - ToTensor (skala [0,1])
    - Normalize dengan mean & std dari checkpoint
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    device   = next(model.parameters()).device
    tensor   = transform(image.convert('RGB')).unsqueeze(0).to(device)
    logits   = model(tensor)
    probs    = F.softmax(logits, dim=1).squeeze(0)
    probs_np = probs.cpu().numpy()
    pred_idx = int(probs.argmax().item())
    return probs_np, pred_idx