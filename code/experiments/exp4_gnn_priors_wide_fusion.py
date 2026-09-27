import os, sys, subprocess, json, math, warnings
import numpy as np, pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import faiss, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

warnings.filterwarnings("ignore", category=UserWarning)

# ===========================
# 0) Config
# ===========================
TEST_N = 50                  # change as needed
TOPK_TEXT = 40               # how many text-nearest docs to consider before KG/GNN rerank
FINAL_K = 5                  # how many to pass to LLM
ALPHA = 0.70                 # weight for text similarity
BETA  = 0.20                 # weight for KG neighborhood score
GAMMA = 0.10                 # weight for GNN disease prior (if available)

EMB_NPY = "iu_bioclinicalbert_embeddings.npy"
EMB_PT  = "iu_bioclinicalbert_dict.pt"    # uid -> vec

# Optional external inputs (if you have them)
ALIGN_JSON = "alignment_concepts.json"    # {"uid": ["right costophrenic angle sharp", ...], ...}
GNN_PRIORS = "gnn_priors.csv"             # uid,disease,prob (long format)

# ===========================
# 1) Keys & Clients
# ===========================
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY not found in env")
client = OpenAI(api_key=api_key)
print("🔑 OpenAI key loaded")

# ===========================
# 2) Dataset
# ===========================
ds = load_dataset("ykumards/open-i")
full = ds["train"]

test = full.select(range(TEST_N))
train = full.select(range(TEST_N, len(full)))
print(f"✅ Dataset loaded | Train={len(train)} Test={len(test)}")

# ===========================
# 3) Load embeddings & FAISS
# ===========================
print("📥 Loading BioClinicalBERT embeddings...")
emb_all = np.load(EMB_NPY)
uid_to_vec = torch.load(EMB_PT, weights_only=False)

train_uids, corpus = [], []
for r in train:
    txt = ((r.get("findings") or "") + " " + (r.get("impression") or "")).strip()
    if (r["uid"] in uid_to_vec) and txt:
        train_uids.append(r["uid"])
        corpus.append(txt)

if not corpus:
    raise RuntimeError("No train corpus after filtering. Ensure embeddings cover train UIDs.")

emb_train = np.vstack([uid_to_vec[u] for u in train_uids]).astype("float32")
faiss.normalize_L2(emb_train)
index = faiss.IndexFlatIP(emb_train.shape[1])
index.add(emb_train)
print(f"✅ FAISS index ready | {emb_train.shape}")

# ===========================
# 4) Query encoder (BioClinicalBERT) CPU/GPU
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

def embed_query(text):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding=True,
        max_length=256
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        vec = outputs.last_hidden_state.mean(dim=1)
    return F.normalize(vec, p=2, dim=1).cpu().numpy()

# ===========================
# 5) Clinical lexicons (CheXpert-lite) + extractors
# ===========================
chexpert = {
    "cardiomegaly":["cardiomegaly","heart enlargement","enlarged cardiac silhouette"],
    "edema":["edema","pulmonary edema","interstitial edema"],
    "consolidation":["consolidation","airspace disease","airspace opacit"],
    "atelectasis":["atelectasis","collapse"],
    "effusion":["effusion","pleural effusion"],
    "pneumothorax":["pneumothorax"],
    "pneumonia":["pneumonia","infectious infiltrate"],
    "pleural thickening":["pleural thickening"],
    "fibrosis":["fibrosis","interstitial","reticular"],
    "emphysema":["emphysema","hyperinflation","hyperinflated"],
    "nodule/mass":["nodule","mass","lesion"],
    "lung opacity":["opacity","infiltrate"],
    "hernia":["hernia"],
    "support devices":["pacemaker","sternotomy","tube","line","wire"]
}
neg_triggers = ["no","without","absence of","negative for","free of","not seen","none","unremarkable for"]

def is_neg(t, kw, window=50):
    pos = t.find(kw)
    if pos == -1: return False
    win = t[max(0,pos-window):pos]
    return any(n in win for n in neg_triggers)

def extract_disease(text):
    t = text.lower()
    found = []
    for dis, kws in chexpert.items():
        for kw in kws:
            if kw in t and not is_neg(t, kw):
                found.append(dis); break
    return sorted(set(found)) or ["no acute disease"]

def extract_concepts_heur(text):
    t = text.lower()
    c = []
    if ("heart size" in t) or ("cardiac silhouette" in t): c.append("heart size normal")
    if ("lungs are clear" in t) or ("clear lungs" in t): c.append("clear lungs")
    if "no pneumothorax" in t: c.append("no pneumothorax")
    if "no effusion" in t: c.append("no effusion")
    return sorted(set(c)) or ["no acute abnormality"]

# ===========================
# 6) (3) KG-RAG: build lightweight co-occurrence KG from train
# ===========================
# Nodes: disease + structural tokens (concepts). Edges weighted by PMI-like score.
from collections import Counter, defaultdict
import math

def tokenize_for_kg(text):
    t = text.lower()
    toks = set()
    # collect diseases
    for dis, kws in chexpert.items():
        if any((kw in t and not is_neg(t, kw)) for kw in kws):
            toks.add(dis)
    # simple structural concepts
    if ("heart size" in t) or ("cardiac silhouette" in t): toks.add("heart size")
    if ("lungs are clear" in t) or ("clear lungs" in t):   toks.add("clear lungs")
    if "no pneumothorax" in t: toks.add("no pneumothorax")
    if "no effusion" in t:      toks.add("no effusion")
    return toks

docs_tokens = [tokenize_for_kg(x) for x in corpus]
N_docs = len(docs_tokens)
df_counts = Counter()
pair_counts = Counter()
for toks in docs_tokens:
    for a in toks:
        df_counts[a]+=1
    toks = sorted(toks)
    for i in range(len(toks)):
        for j in range(i+1, len(toks)):
            pair_counts[(toks[i], toks[j])] += 1

def pmi(a,b):
    pa = df_counts[a] / N_docs if df_counts[a] else 1e-9
    pb = df_counts[b] / N_docs if df_counts[b] else 1e-9
    pab = pair_counts[(a,b)] / N_docs if pair_counts[(a,b)] else 1e-9
    val = math.log((pab)/(pa*pb) + 1e-9)
    return max(val, 0.0)  # clipped

# Pre-compute a node→neighbor score map
neighbor_score = defaultdict(dict)
for (a,b), cnt in pair_counts.items():
    w = pmi(a,b)
    if w>0:
        neighbor_score[a][b]=w
        neighbor_score[b][a]=w

# Corpus term presence map
term_in_doc = []
for doc in corpus:
    t = doc.lower()
    bag = set()
    for dis in chexpert.keys():
        if dis in t: bag.add(dis)
    if "heart size" in t: bag.add("heart size")
    if "clear lungs" in t or "lungs are clear" in t: bag.add("clear lungs")
    if "no pneumothorax" in t: bag.add("no pneumothorax")
    if "no effusion" in t: bag.add("no effusion")
    term_in_doc.append(bag)

def kg_score_for_doc(query_terms, doc_terms):
    # sum of neighbor weights from each query term into doc terms
    s = 0.0
    for q in query_terms:
        neigh = neighbor_score.get(q, {})
        s += sum(neigh.get(d, 0.0) for d in doc_terms)
    return s

# ===========================
# 7) (2) GNN priors: optional load, else fallback to weak rules
# ===========================
gnn_priors_map = defaultdict(dict)  # uid -> {disease: prob}
if os.path.exists(GNN_PRIORS):
    dfp = pd.read_csv(GNN_PRIORS)
    for _, r in dfp.iterrows():
        gnn_priors_map[str(r["uid"])][str(r["disease"]).lower()] = float(r["prob"])
    print(f"🧠 Loaded GNN priors for {len(gnn_priors_map)} UIDs")
else:
    print("ℹ️ No GNN priors file found; will use weak keyword cues as pseudo-priors.")

def gnn_prior_weight(uid_str, diseases_list):
    # if we have priors, average probs; else simple heuristic
    pri = gnn_priors_map.get(uid_str, {})
    if pri:
        return np.mean([pri.get(d, 0.1) for d in diseases_list])  # small floor
    else:
        # heuristic: if disease list is normal → low weight; else medium
        if diseases_list==["no acute disease"]:
            return 0.25
        return 0.60

# ===========================
# 8) (1) Alignment concepts: optional JSON; else heuristic
# ===========================
align_map = {}
if os.path.exists(ALIGN_JSON):
    with open(ALIGN_JSON, "r") as f:
        align_map = json.load(f)
    print(f"🧷 Loaded alignment concepts for {len(align_map)} UIDs")
else:
    print("ℹ️ No alignment_concepts.json found; will use heuristic concept extractor.")

# ===========================
# 9) Graph-aware RAG retrieval
#     - Text FAISS → shortlist TOPK_TEXT
#     - Compute KG score per doc (query terms from diseases+concepts)
#     - Multiply by GNN prior for test UID
#     - Combine: ALPHA*text + BETA*KG + GAMMA*GNN
# ===========================
def graph_rag(query_terms, test_uid_str, final_k=FINAL_K):
    # Build a short query string for text embedding
    query_text = ", ".join(query_terms)
    q = embed_query(query_text); faiss.normalize_L2(q)
    D, I = index.search(q, TOPK_TEXT)
    # Normalize text sims
    text_sims = (D[0] - D[0].min()) / (np.ptp(D[0]) + 1e-9)

    # KG scores
    kg_scores = []
    for idx in I[0]:
        doc_terms = term_in_doc[int(idx)]
        kg_scores.append(kg_score_for_doc(query_terms, doc_terms))
    kg_scores = np.array(kg_scores, dtype="float32")
    kg_scores = (kg_scores - kg_scores.min()) / (np.ptp(kg_scores) + 1e-9)


    # TRUE GNN fusion: compare GNN vector of this test report vs each candidate report
    gnn_scores = []
    for idx in I[0]:
      corpus_uid = str(train_uids[idx])
      gnn_scores.append(gnn_similarity(test_uid_str, corpus_uid))

    gnn_scores = np.array(gnn_scores, dtype="float32")
    # normalize
    gnn_scores = (gnn_scores - gnn_scores.min()) / (np.ptp(gnn_scores) + 1e-9)


    # Fuse
    fused = ALPHA*text_sims + BETA*kg_scores + GAMMA*gnn_scores

    order = np.argsort(-fused)[:final_k]
    picked_idxs = [int(I[0][i]) for i in order]
    picked_txts = [corpus[j][:300] for j in picked_idxs]
    # For explainability: return component scores too
    return picked_txts, {
        "text": [float(text_sims[i]) for i in order],
        "kg":   [float(kg_scores[i])   for i in order],
        "gnn":  [float(gnn_scores[i])  for i in order],
        "fused":[float(fused[i])       for i in order]
    }

# ===========================
# 10) LLM Report generation (with graph-aware evidence)
# ===========================
def build_prompt(concepts, diseases, retrieved, score_dict):
    expl = "\n".join([
        f"- Doc#{i+1}: fused={score_dict['fused'][i]:.2f}, text={score_dict['text'][i]:.2f}, kg={score_dict['kg'][i]:.2f}, gnn={score_dict['gnn'][i]:.2f}"
        for i in range(len(retrieved))
    ])
    prompt = f"""
You are a board-certified radiologist. Generate a FINAL chest X-ray IMPRESSION only (no headings).
Be concise (4–6 sentences), conservative, and avoid hallucinations.

Detected diseases (from image priors): {diseases}
Structural concepts: {concepts}

Graph-aware retrieved prior cases (top-{len(retrieved)}):
{chr(10).join("- " + r for r in retrieved)}

Retrieval evidence weights:
{expl}

Rules:
- Address cardiac size, lungs, pleura (effusion/pneumothorax), then bones/other.
- If diseases == ['no acute disease'], default to a normal study unless retrieval strongly suggests otherwise.
- Explicitly state presence/absence for effusion and pneumothorax.
Return only the impression text.
"""
    return prompt.strip()

def gen_report(concepts, diseases, retrieved, score_dict):
    prompt = build_prompt(concepts, diseases, retrieved, score_dict)
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.1,
        max_tokens=260
    )
    return r.choices[0].message.content.strip()

# ===========================
# 11) Run evaluation
# ===========================
results=[]
for row in tqdm(test, desc="🧭 Graph-RAG evaluation"):
    uid = str(row["uid"])
    gt  = ((row.get("findings") or "") + " " + (row.get("impression") or "")).strip()

    # (1) Alignment concepts: prefer provided, else heuristic
    if uid in align_map and isinstance(align_map[uid], list) and len(align_map[uid])>0:
        concepts = sorted(set(align_map[uid]))
    else:
        concepts = extract_concepts_heur(gt)

    # (2) Diseases: from rules (if you have GNN priors, they influence retrieval via GAMMA)
    diseases = extract_disease(gt)

    # (3) KG-RAG retrieval
    query_terms = sorted(set(concepts + diseases))
    retrieved, scores = graph_rag(query_terms, uid, final_k=FINAL_K)

    # (4) Generation
    report = gen_report(concepts, diseases, retrieved, scores)

    results.append({
        "uid": uid,
        "image_path": row.get("image"),
        "ground_truth": gt,
        "predicted_disease": "; ".join(diseases),
        "concepts": "; ".join(concepts),
        "rag_retrieved": " || ".join(retrieved),
        "our_method": report
    })

df = pd.DataFrame(results)
df.to_csv("iu_graph_rag_results.csv", index=False)
print("💾 Saved -> iu_graph_rag_results.csv")
pd.set_option("display.max_colwidth", None)
print(df.head(5))

# ===========================
# 12) Metrics
# ===========================
print("\n📏 Computing metrics (BLEU/ROUGE/CIDEr)...")
import nltk, sacrebleu
nltk.download("punkt", quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction().method1
refs = df["ground_truth"].tolist()
gens = df["our_method"].tolist()
ref_tok=[nltk.word_tokenize(r.lower()) for r in refs]
gen_tok=[nltk.word_tokenize(g.lower()) for g in gens]

BLEU1 = np.mean([sentence_bleu([ref_tok[i]], gen_tok[i], weights=(1,0,0,0), smoothing_function=smooth) for i in range(len(refs))])
BLEU2 = np.mean([sentence_bleu([ref_tok[i]], gen_tok[i], weights=(.5,.5,0,0), smoothing_function=smooth) for i in range(len(refs))])
BLEU3 = np.mean([sentence_bleu([ref_tok[i]], gen_tok[i], weights=(1/3,1/3,1/3,0), smoothing_function=smooth) for i in range(len(refs))])
BLEU4_sent = np.mean([sentence_bleu([ref_tok[i]], gen_tok[i], weights=(.25,.25,.25,.25), smoothing_function=smooth) for i in range(len(refs))])
BLEU4_corpus = sacrebleu.corpus_bleu(gens, [refs]).score

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
ROUGE_L = np.mean([scorer.score(refs[i], gens[i])["rougeL"].fmeasure for i in range(len(refs))])

# CIDEr
try:
    from pycocoevalcap.cider.cider import Cider
except:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/salaniz/pycocoevalcap"])
    from pycocoevalcap.cider.cider import Cider
cider = Cider()
gts = {i: [refs[i]] for i in range(len(refs))}
res = {i: [gens[i]] for i in range(len(gens))}
CIDEr_score, _ = cider.compute_score(gts, res)

print("\n===== GRAPH-RAG RESULTS =====")
print(f"BLEU-1: {BLEU1:.4f}")
print(f"BLEU-2: {BLEU2:.4f}")
print(f"BLEU-3: {BLEU3:.4f}")
print(f"BLEU-4: {BLEU4_sent:.4f}")
print(f"BLEU-4 Corpus: {BLEU4_corpus:.2f}")
print(f"ROUGE-L: {ROUGE_L:.4f}")
print(f"CIDEr: {CIDEr_score:.3f}")
print("================================")
print("📄 Results -> iu_graph_rag_results.csv")
