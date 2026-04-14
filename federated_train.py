import copy
import csv
import datetime
import hashlib
import json
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# =====================
# SETTINGS
# =====================
DATA_PATH = "dataset/bus_uclm_separated"
NUM_CLIENTS = 3
ROUNDS = 5
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-4
TEST_RATIO = 0.3

# Differential privacy settings
LOCAL_DP_STD = 0.01        # Gaussian noise on client updates
GLOBAL_DP_STD = 0.005      # Gaussian noise after aggregation
DP_CLIP_NORM = 1.0         # Norm clipping for update sensitivity

# Secure aggregation / encryption
USE_SECURE_AGGREGATION = True
USE_ENCRYPTION = False     # Optional simulated encryption of client updates
ENCRYPTION_KEY = "secure_key_123"
NUM_SHARES = NUM_CLIENTS   # Number of additive shares in secure aggregation

# Output settings
RESULTS_DIR = "results"
RESULTS_FILE = "federated_results.csv"
RESULTS_PATH = os.path.join(RESULTS_DIR, RESULTS_FILE)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("secure_federated_train")
logger.info("Using device: %s", device)

# =====================
# DATA TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =====================
# DATA PREPARATION
# =====================
def prepare_dataset(data_path):
    dataset = ImageFolder(data_path, transform=transform)
    filtered = [
        (path, label) for path, label in dataset.samples
        if dataset.classes[label] in ["benign", "malign"]
    ]
    dataset.samples = filtered
    dataset.targets = [0 if dataset.classes[label] == "benign" else 1 for _, label in filtered]
    dataset.classes = ["benign", "malign"]
    dataset.class_to_idx = {"benign": 0, "malign": 1}
    logger.info("Filtered dataset to binary labels: %d samples", len(dataset))
    return dataset


def split_dataset(dataset, num_clients, test_ratio):
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    _, test_dataset = random_split(dataset, [train_size, test_size])

    base = len(dataset) // num_clients
    remainder = len(dataset) - base * num_clients
    client_sizes = [base + (1 if i < remainder else 0) for i in range(num_clients)]
    client_datasets = random_split(dataset, client_sizes)
    logger.info("Split dataset into %d clients and %d-test samples", num_clients, len(test_dataset))
    return client_datasets, test_dataset


# =====================
# MODEL DEFINITION
# =====================
class Adapter(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.down = nn.Linear(dim, dim // reduction)
        self.act = nn.GELU()
        self.up = nn.Linear(dim // reduction, dim)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


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


def create_model():
    vit = timm.create_model("vit_base_patch16_224", pretrained=True)
    for p in vit.parameters():
        p.requires_grad = False
    return JUVILImage(vit).to(device)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


# =====================
# PRIVACY UTILITIES
# =====================
def _hash_seed(key: str, client_id: int) -> int:
    digest = hashlib.sha256(f"{key}-{client_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def add_gaussian_noise(tensor, std):
    if std <= 0:
        return tensor
    return tensor + torch.normal(0, std, size=tensor.shape, device=tensor.device)


def compute_update(global_state, local_state):
    return {
        key: local_state[key].float() - global_state[key].float()
        for key in global_state.keys()
    }


def clip_update(update, clip_norm):
    squared_norm = sum((tensor ** 2).sum() for tensor in update.values())
    total_norm = torch.sqrt(squared_norm)
    if total_norm <= clip_norm or total_norm == 0.0:
        return update
    scale = clip_norm / (total_norm + 1e-12)
    return {key: tensor * scale for key, tensor in update.items()}


def apply_local_dp(update, sigma, clip_norm):
    clipped = clip_update(update, clip_norm)
    return {
        key: add_gaussian_noise(tensor, sigma * clip_norm)
        for key, tensor in clipped.items()
    }


def _encryption_mask(shape, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(shape, device=device, generator=gen)


def encrypt_update(update, key, client_id):
    encrypted = {}
    for key_name, tensor in update.items():
        seed = _hash_seed(key, client_id + hash(key_name))
        encrypted[key_name] = tensor + _encryption_mask(tensor.shape, seed)
    return encrypted


def decrypt_update(encrypted_update, key, client_id):
    decrypted = {}
    for key_name, tensor in encrypted_update.items():
        seed = _hash_seed(key, client_id + hash(key_name))
        decrypted[key_name] = tensor - _encryption_mask(tensor.shape, seed)
    return decrypted


def share_update(update, num_shares):
    shares = [dict() for _ in range(num_shares)]
    for key, tensor in update.items():
        partials = [torch.randn_like(tensor) for _ in range(num_shares - 1)]
        last_share = tensor - sum(partials)
        for index, share in enumerate(partials):
            shares[index][key] = share
        shares[-1][key] = last_share
    return shares


def reconstruct_from_shares(shares):
    return {
        key: sum(share[key] for share in shares)
        for key in shares[0].keys()
    }


def secure_aggregate(updates, num_shares):
    logger.info("Using secure aggregation with %d additive shares", num_shares)
    all_client_shares = [share_update(update, num_shares) for update in updates]
    aggregated_shares = []
    for share_index in range(num_shares):
        aggregated_shares.append(reconstruct_from_shares(
            [client_share[share_index] for client_share in all_client_shares]
        ))
    return reconstruct_from_shares(aggregated_shares)


def aggregate_updates(updates, num_clients, use_secure, global_dp_std, clip_norm):
    if use_secure:
        summed_updates = secure_aggregate(updates, NUM_SHARES)
    else:
        summed_updates = {
            key: sum(update[key] for update in updates)
            for key in updates[0].keys()
        }
    averaged = {key: tensor / num_clients for key, tensor in summed_updates.items()}
    if global_dp_std > 0:
        averaged = {
            key: add_gaussian_noise(tensor, global_dp_std * clip_norm)
            for key, tensor in averaged.items()
        }
        logger.info("Applied global DP Gaussian noise with sigma=%s", global_dp_std)
    return averaged


def apply_update_to_model(model, update):
    state = model.state_dict()
    for key in state.keys():
        state[key] = state[key].float() + update[key]
    model.load_state_dict(state)


def save_result_row(path, row, write_header=False):
    mode = "w" if write_header else "a"
    with open(path, mode, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_summary(summary_path, html_path, summary):
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(html_path, "w") as f:
        f.write("<html><head><meta charset=\"utf-8\"><title>Federated Results</title></head><body>")
        f.write("<h2>Federated Training Results</h2>")
        if summary["best_round"]:
            best = summary["best_round"]
            f.write(f"<p><strong>Best round:</strong> {best['round']} — accuracy: {best['accuracy']}</p>")
        f.write("<table border=\"1\"><tr><th>timestamp</th><th>round</th><th>accuracy</th><th>local_dp_std</th><th>global_dp_std</th><th>secure_agg</th><th>encryption</th></tr>")
        for r in summary["rounds"]:
            f.write(
                f"<tr><td>{r['timestamp']}</td><td>{r['round']}</td><td>{r['accuracy']}</td>"
                f"<td>{r['local_dp_std']}</td><td>{r['global_dp_std']}</td>"
                f"<td>{r['secure_aggregation']}</td><td>{r['encryption']}</td></tr>"
            )
        f.write("</table></body></html>")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = prepare_dataset(DATA_PATH)
    client_datasets, test_dataset = split_dataset(dataset, NUM_CLIENTS, TEST_RATIO)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    if not os.path.exists(RESULTS_PATH):
        save_result_row(RESULTS_PATH, [
            "timestamp", "round", "accuracy", "local_dp_std", "global_dp_std",
            "secure_aggregation", "encryption"
        ], write_header=True)

    global_model = create_model()
    results_list = []

    for round_index in range(ROUNDS):
        logger.info("Starting federated round %d/%d", round_index + 1, ROUNDS)
        client_updates = []

        for client_id in range(NUM_CLIENTS):
            logger.info("Client %d local training", client_id + 1)
            client_model = copy.deepcopy(global_model)
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, client_model.parameters()),
                lr=LR
            )
            criterion = nn.CrossEntropyLoss()
            loader = DataLoader(client_datasets[client_id], batch_size=BATCH_SIZE, shuffle=True)

            client_model.train()
            for _ in range(LOCAL_EPOCHS):
                for imgs, labels in loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out = client_model(imgs)
                    loss = criterion(out, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            local_update = compute_update(global_model.state_dict(), client_model.state_dict())
            local_update = apply_local_dp(local_update, LOCAL_DP_STD, DP_CLIP_NORM)
            logger.info("Applied local DP on client %d updates", client_id + 1)

            if USE_ENCRYPTION:
                local_update = encrypt_update(local_update, ENCRYPTION_KEY, client_id)
                logger.info("Encrypted client %d update", client_id + 1)

            client_updates.append(local_update)

        if USE_ENCRYPTION:
            client_updates = [decrypt_update(update, ENCRYPTION_KEY, client_id) for client_id, update in enumerate(client_updates)]
            logger.info("Decrypted encrypted client updates before aggregation")

        aggregated_update = aggregate_updates(
            client_updates,
            NUM_CLIENTS,
            USE_SECURE_AGGREGATION,
            GLOBAL_DP_STD,
            DP_CLIP_NORM,
        )
        apply_update_to_model(global_model, aggregated_update)

        accuracy = evaluate(global_model, test_loader)
        logger.info("Round %d evaluation accuracy: %.4f", round_index + 1, accuracy)

        row = [
            datetime.datetime.now().isoformat(),
            round_index + 1,
            float(f"{accuracy:.4f}"),
            LOCAL_DP_STD,
            GLOBAL_DP_STD,
            USE_SECURE_AGGREGATION,
            USE_ENCRYPTION,
        ]
        save_result_row(RESULTS_PATH, row)
        results_list.append({
            "timestamp": row[0],
            "round": row[1],
            "accuracy": row[2],
            "local_dp_std": row[3],
            "global_dp_std": row[4],
            "secure_aggregation": row[5],
            "encryption": row[6],
        })

    torch.save(global_model.state_dict(), "federated_model.pth")
    logger.info("Federated model saved at federated_model.pth")

    summary = {
        "privacy_config": {
            "local_dp_std": LOCAL_DP_STD,
            "global_dp_std": GLOBAL_DP_STD,
            "dp_clip_norm": DP_CLIP_NORM,
            "secure_aggregation": USE_SECURE_AGGREGATION,
            "encryption": USE_ENCRYPTION,
            "num_clients": NUM_CLIENTS,
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
        },
        "best_round": max(results_list, key=lambda x: x["accuracy"])
        if results_list
        else None,
        "rounds": results_list,
    }

    summary_path = os.path.join(RESULTS_DIR, "federated_results_summary.json")
    html_path = os.path.join(RESULTS_DIR, "federated_results.html")
    save_summary(summary_path, html_path, summary)
    logger.info("Results written: %s, %s", RESULTS_PATH, summary_path)


if __name__ == "__main__":
    main()
