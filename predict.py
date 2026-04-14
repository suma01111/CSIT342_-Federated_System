import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

# =====================
# CHANGE THESE
# =====================
# MODEL_PATH = "breast_cancer_model.pth"   # or federated_model.pth
MODEL_PATH = "breast_cancer_model.pth"   # or federated_model.pth
IMAGE_PATH = "dataset/bus_uclm_separated/malign/CHCO_000.png"  # any image

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =====================
# ADAPTER
# =====================
class Adapter(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.down = nn.Linear(dim, dim // reduction)
        self.act = nn.GELU()
        self.up = nn.Linear(dim // reduction, dim)

    def forward(self, x):
        return self.up(self.act(self.down(x)))

# =====================
# MODEL STRUCTURE
# =====================
vit = timm.create_model("vit_base_patch16_224", pretrained=False)

for p in vit.parameters():
    p.requires_grad = False

class JUVILImage(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        dim = vit.embed_dim
        self.adapter = Adapter(dim)
        self.head = nn.Linear(dim, 2)

    def forward(self, x):
        feat = self.vit.forward_features(x)
        if len(feat.shape) == 3:
            feat = feat[:, 0]
        feat = feat + self.adapter(feat)
        return self.head(feat)

model = JUVILImage(vit).to(device)

# =====================
# LOAD TRAINED WEIGHTS
# =====================
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =====================
# LOAD IMAGE
# =====================
img = Image.open(IMAGE_PATH).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

# =====================
# PREDICTION
# =====================
with torch.no_grad():
    out = model(img)
    prob = torch.softmax(out, dim=1)
    pred = prob.argmax(1).item()
    confidence = prob[0][pred].item()

classes = ["Benign", "Malignant"]

print("Prediction:", classes[pred])
print("Confidence:", round(confidence, 4))
