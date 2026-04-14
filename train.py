import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef

import timm

# =====================
# SETTINGS
# =====================
DATA_PATH = "dataset/bus_uclm_separated"
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =====================
# IMAGE TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =====================
# LOAD DATASET
# =====================
dataset = ImageFolder(DATA_PATH, transform=transform)

# remove NORMAL
filtered_samples = [
    (path, label) for path, label in dataset.samples
    if dataset.classes[label] in ['benign', 'malign']
]

dataset.samples = filtered_samples
dataset.targets = [0 if dataset.classes[label]=='benign' else 1 for _, label in filtered_samples]
dataset.classes = ['benign', 'malign']
dataset.class_to_idx = {'benign': 0, 'malign': 1}

print("Total images:", len(dataset))

# =====================
# SPLIT TRAIN / TEST
# =====================
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

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
# MODEL
# =====================
vit = timm.create_model("vit_base_patch16_224", pretrained=True)

# freeze backbone
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
# LOSS & OPTIMIZER
# =====================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# =====================
# TRAINING
# =====================
for epoch in range(EPOCHS):
    model.train()
    correct = 0
    total = 0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        out = model(imgs)
        loss = criterion(out, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = out.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    print(f"Epoch {epoch+1}/{EPOCHS} Train Acc: {correct/total:.4f}")

# =====================
# TESTING
# =====================
model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        out = model(imgs)
        
        prob = torch.softmax(out, dim=1)[:,1]
        pred = out.argmax(1).cpu()
        
        y_true.extend(labels.numpy())
        y_pred.extend(pred.numpy())
        y_prob.extend(prob.cpu().numpy())

acc = sum([a==b for a,b in zip(y_true,y_pred)]) / len(y_true)
auc = roc_auc_score(y_true, y_prob)
mcc = matthews_corrcoef(y_true, y_pred)

print("\nFINAL RESULTS")
print("Accuracy:", acc)
print("AUC:", auc)
print("MCC:", mcc)

torch.save(model.state_dict(), "breast_cancer_model.pth")
print("Model saved!")
