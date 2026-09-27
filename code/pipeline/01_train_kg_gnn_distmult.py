import os, json, math, random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# -----------------------------
# 1) Load IU dataset & split
# -----------------------------
TEST_N = 50  # keep same as your RAG experiments
ds = load_dataset("ykumards/open-i")
full = ds["train"]

test = full.select(range(TEST_N))
train = full.select(range(TEST_N, len(full)))

def get_text(r):
    return ((r.get("findings") or "") + " " + (r.get("impression") or "")).strip()

# -----------------------------
# 2) CheXpert-like disease rules
# -----------------------------
chexpert = {
    "cardiomegaly": ["cardiomegaly", "heart enlargement", "enlarged cardiac silhouette"],
    "edema": ["edema", "pulmonary edema", "interstitial edema"],
    "consolidation": ["consolidation", "airspace disease", "airspace opacit"],
    "atelectasis": ["atelectasis", "collapse"],
    "effusion": ["pleural effusion", "effusion"],
    "pneumothorax": ["pneumothorax"],
    "pneumonia": ["pneumonia", "infectious infiltrate"],
    "pleural thickening": ["pleural thickening"],
    "fibrosis": ["fibrosis", "interstitial lung disease", "reticular"],
    "emphysema": ["emphysema", "hyperinflation", "hyperinflated"],
    "nodule/mass": ["nodule", "mass", "lesion"],
    "lung opacity": ["opacity", "infiltrate"],
    "hernia": ["hernia"],
    "support devices": ["pacemaker", "sternotomy", "tube", "line", "wire"],
}
neg_triggers = ["no", "without", "absence of", "negative for", "free of", "not seen", "none", "unremarkable for"]

def is_neg(t, kw, window=50):
    pos = t.find(kw)
    if pos == -1: return False
    win = t[max(0,pos-window):pos]
    return any(n in win for n in neg_triggers)

def extract_diseases(text):
    t = text.lower()
    out = []
    for dis, kws in chexpert.items():
        for kw in kws:
            if kw in t and not is_neg(t, kw):
                out.append(dis); break
    return sorted(set(out))

# -----------------------------
# 3) Build KG entities & triples
# Entities = report nodes + disease nodes
# Relation = 'indicates' (single relation)
# -----------------------------
report_uids_train, triples_pos = [], []
disease_vocab = sorted(list(chexpert.keys()))

# Map diseases to contiguous IDs (they will be entity nodes too)
disease2eid = {d: i for i, d in enumerate(disease_vocab)}  # disease entity ids [0..D-1]

# Reports will be after diseases in the entity table
def build_triples(split):
    triples = []
    r_uids = []
    for r in split:
        uid = str(r["uid"])
        txt = get_text(r)
        labs = extract_diseases(txt)
        r_uids.append(uid)
        for d in labs:
            triples.append((uid, d))  # (report uid, disease)
    return r_uids, triples

train_uids, train_pairs = build_triples(train)
test_uids,  test_pairs  = build_triples(test)

# Keep only reports that have at least one positive label in train
train_uids_unique = sorted(set(train_uids))
# build report entity ids
report2eid = {uid: (len(disease2eid) + i) for i, uid in enumerate(train_uids_unique)}
num_disease = len(disease2eid)
num_report  = len(report2eid)
num_entities = num_disease + num_report

print(f"Entities: diseases={num_disease}, reports(train)={num_report}, total={num_entities}")

# Convert positives to (h, r, t) with r=0 always (single relation)
REL_INDICATES = 0
pos_triples = []
for uid, dis in train_pairs:
    if uid in report2eid:
        h = report2eid[uid]
        t = disease2eid[dis]
        pos_triples.append((h, REL_INDICATES, t))

print(f"Train positive triples: {len(pos_triples)}")

# -----------------------------
# 4) DistMult model & training
# -----------------------------
class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations=1, dim=128):
        super().__init__()
        self.dim = dim
        self.ent = nn.Embedding(num_entities, dim)
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def score(self, h_idx, r_idx, t_idx):
        # DistMult: sum(e_h * r * e_t)
        h = self.ent(h_idx)          # [B, dim]
        r = self.rel(r_idx)          # [B, dim]
        t = self.ent(t_idx)          # [B, dim]
        return torch.sum(h * r * t, dim=-1)  # [B]

    def forward(self, h_idx, r_idx, t_idx):
        return self.score(h_idx, r_idx, t_idx)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DistMult(num_entities=num_entities, num_relations=1, dim=128).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

def sample_batch(pos_triples, batch_size=2048, neg_ratio=1.0):
    # Random positives
    pos_idx = np.random.randint(0, len(pos_triples), size=min(batch_size, len(pos_triples)))
    batch_pos = [pos_triples[i] for i in pos_idx]

    # Negative sampling by corrupting tail (disease)
    negs = []
    for (h, r, t) in batch_pos:
        for _ in range(int(neg_ratio)):
            t_neg = np.random.randint(0, num_disease)  # choose random disease
            negs.append((h, r, t_neg))
    return batch_pos, negs

bce = nn.BCEWithLogitsLoss()

EPOCHS = 40
STEPS_PER_EPOCH = 80
NEG_RATIO = 2
best_loss = 1e9
patience, bad = 6, 0

for epoch in range(1, EPOCHS+1):
    model.train()
    losses = []
    for _ in range(STEPS_PER_EPOCH):
        pos, neg = sample_batch(pos_triples, batch_size=1024, neg_ratio=NEG_RATIO)
        batch = pos + neg
        labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(device)

        h = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
        r = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
        t = torch.tensor([x[2] for x in batch], dtype=torch.long, device=device)

        scores = model(h, r, t)
        loss = bce(scores, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    avg = float(np.mean(losses))
    print(f"[Epoch {epoch:02d}] loss={avg:.4f}")
    if avg < best_loss - 1e-4:
        best_loss = avg
        bad = 0
        torch.save(model.state_dict(), "distmult_iu.ckpt")
    else:
        bad += 1
        if bad >= patience:
            print("Early stopping."); break

# Load best
model.load_state_dict(torch.load("distmult_iu.ckpt", map_location=device))

# -----------------------------
# 5) Inference: per-UID disease probabilities
# For ALL UIDs (train + test), we compute P(disease | uid)
# For train UIDs: they have learned report embeddings
# For test UIDs: cold-start → we use a simple neighbor backoff:
#   score(uid_test, d) = avg score of top-K similar train reports' disease scores
#   Similarity: Jaccard overlap of rule labels (fast, no extra deps)
# -----------------------------
def jaccard(a, b):
    if not a and not b: return 0.0
    return len(a & b) / (len(a | b) + 1e-9)

# Precompute train label sets
train_uid_to_labels = {}
for r in train:
    uid = str(r["uid"])
    labs = set(extract_diseases(get_text(r)))
    train_uid_to_labels[uid] = labs

# Build dense matrix of train predictions (reports x diseases)
model.eval()
with torch.no_grad():
    # For train reports, compute scores to all diseases
    train_scores = np.zeros((len(train_uids_unique), num_disease), dtype="float32")
    r_idx = torch.tensor([report2eid[u] for u in train_uids_unique], dtype=torch.long, device=device)
    r_vec = model.ent(r_idx)  # [R, dim]
    rel_vec = model.rel(torch.tensor([REL_INDICATES], device=device)).squeeze(0)  # [dim]
    d_idx = torch.arange(0, num_disease, dtype=torch.long, device=device)
    d_vec = model.ent(d_idx)  # [D, dim]
    # score = sum(r * rel * d)
    scores = torch.matmul(r_vec * rel_vec, d_vec.t())  # [R, D]
    train_scores = torch.sigmoid(scores).cpu().numpy()

# Map train uid -> vector
train_uid_to_vec = {u: train_scores[i] for i, u in enumerate(train_uids_unique)}

def predict_probs_for_uid(uid, txt):
    # If uid in training (has embedding), use direct scores
    if uid in train_uid_to_vec:
        return train_uid_to_vec[uid]

    # Else cold-start: label-based KNN over train uids
    labs = set(extract_diseases(txt))
    if len(train_uid_to_vec) == 0:
        return np.full((num_disease,), 0.1, dtype="float32")

    # compute top-K neighbors by Jaccard
    sims = []
    for tr_uid, tr_labs in train_uid_to_labels.items():
        sims.append((tr_uid, jaccard(labs, tr_labs)))
    sims.sort(key=lambda x: x[1], reverse=True)
    K = 8
    top = sims[:K]
    # weighted average of neighbor disease vectors
    num = np.zeros((num_disease,), dtype="float32")
    den = 1e-6
    for tr_uid, s in top:
        num += train_uid_to_vec.get(tr_uid, np.zeros_like(num)) * s
        den += s
    return (num / den).astype("float32")

# Collect predictions for all (train + test)
rows_long = []
uids_all = []
for split_name, split in [("train", train), ("test", test)]:
    for r in split:
        uid = str(r["uid"])
        txt = get_text(r)
        probs = predict_probs_for_uid(uid, txt)
        for d, p in zip(disease_vocab, probs):
            rows_long.append({"uid": uid, "disease": d, "prob": float(p), "split": split_name})
        uids_all.append(uid)

df_long = pd.DataFrame(rows_long)
df_wide = df_long.pivot_table(index=["uid","split"], columns="disease", values="prob").reset_index()

df_long.to_csv("gnn_priors.csv", index=False)
df_wide.to_csv("gnn_priors_wide.csv", index=False)

# Save mappings (for reproducibility)
meta = {
    "disease_vocab": disease_vocab,
    "disease2eid": disease2eid,
    "report2eid_size": len(report2eid),
    "relation": {"indicates": REL_INDICATES},
    "seed": SEED,
    "test_n": TEST_N
}
with open("gnn_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Saved:")
print(" - distmult_iu.ckpt")
print(" - gnn_priors.csv        (long: uid,disease,prob,split)")
print(" - gnn_priors_wide.csv   (wide: 1 row per uid)")
print(" - gnn_meta.json")
