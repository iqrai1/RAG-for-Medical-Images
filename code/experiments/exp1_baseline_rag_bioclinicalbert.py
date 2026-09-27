import os, sys, subprocess
import numpy as np, pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import faiss, torch
import torch.nn.functional as F
from openai import OpenAI

# ===========================
# ✅ OpenAI Key
# ===========================
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY not found")
client = OpenAI(api_key=api_key)
print("🔑 OpenAI key loaded")

# ===========================
# ✅ Load IU Dataset
# ===========================
ds = load_dataset("ykumards/open-i")
full = ds["train"]
TEST_N = 50
test = full.select(range(TEST_N))
train = full.select(range(TEST_N, len(full)))
train_uids = [r["uid"] for r in train]
print(f"Dataset loaded ✅ Train={len(train)} Test={len(test)}")

# ===========================
# ✅ Load Precomputed BioClinicalBERT Embeddings
# ===========================
print("📥 Loading saved BioClinicalBERT embeddings...")
emb_all = np.load("iu_bioclinicalbert_embeddings.npy")
uid_to_vec = torch.load("iu_bioclinicalbert_dict.pt", weights_only=False)

valid_uids = []
emb_vectors = []

for uid in train_uids:
    if uid in uid_to_vec:
        valid_uids.append(uid)
        emb_vectors.append(uid_to_vec[uid])
    else:
        # Skip missing embeddings
        continue

emb_train = np.vstack(emb_vectors).astype("float32")
train_uids = valid_uids  # update list so FAISS index matches

faiss.normalize_L2(emb_train)
index = faiss.IndexFlatIP(emb_train.shape[1])
index.add(emb_train)
print(f"FAISS index ready ✅ Shape: {emb_train.shape}")

# ===========================
# ✅ Load BioClinicalBERT for Query Embedding
# ===========================
from transformers import AutoTokenizer, AutoModel

print("🧠 Loading BioClinicalBERT model for query encoding...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

def embed_query(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        vec = outputs.last_hidden_state.mean(dim=1)

    return F.normalize(vec, p=2, dim=1).cpu().numpy()


# ===========================
# ✅ Disease & Concept Extractors
# ===========================
chexpert = {
    "cardiomegaly":["cardiomegaly","heart enlargement","enlarged cardiac silhouette"],
    "edema":["edema","pulmonary edema"],
    "consolidation":["consolidation","airspace"],
    "atelectasis":["atelectasis"],
    "effusion":["effusion","pleural effusion"],
    "pneumothorax":["pneumothorax"],
    "pneumonia":["pneumonia","infectious infiltrate"],
    "pleural thickening":["pleural thickening"],
    "fibrosis":["fibrosis","interstitial"],
    "emphysema":["emphysema","hyperinflated"],
    "nodule/mass":["nodule","mass","lesion"],
    "lung opacity":["opacity","infiltrate"],
    "hernia":["hernia"],
    "support devices":["pacemaker","sternotomy","tube"]
}
neg_triggers=["no","without","absence of","free of","not seen","none"]

def is_neg(text, kw):
    t=text.lower()
    pos=t.find(kw)
    if pos==-1: return False
    win=t[max(0,pos-40):pos]
    return any(n in win for n in neg_triggers)

def extract_disease(text):
    t=text.lower()
    out=[]
    for dis,kws in chexpert.items():
        for kw in kws:
            if kw in t and not is_neg(t,kw): out.append(dis)
    return sorted(set(out)) or ["no acute disease"]

def extract_concepts(text):
    t = text.lower()
    c=[]
    if "heart size" in t or "cardiac silhouette" in t: c.append("heart size normal")
    if "clear lungs" in t or "lungs are clear" in t: c.append("clear lungs")
    if "no pneumothorax" in t: c.append("no pneumothorax")
    if "no effusion" in t: c.append("no effusion")
    return sorted(set(c)) or ["no acute abnormality"]

# ===========================
# ✅ RAG Search
# ===========================
corpus = []
for r in train:
    if r["uid"] in train_uids:  # only include ones with embeddings
        txt = ((r.get("findings") or "") + " " + (r.get("impression") or "")).strip()
        corpus.append(txt)


def rag(query, k=5):
    q = embed_query(query)
    faiss.normalize_L2(q)
    D, I = index.search(q, k*2)

    ranked=[]
    toks=[x.strip() for x in query.lower().split(",")]

    for idx in I[0]:
        ref = corpus[int(idx)]
        score = sum(1 for tok in toks if tok and tok in ref.lower())
        ranked.append((ref, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return [r[0][:300] for r in ranked[:k]]

# ===========================
# ✅ Report Generation
# ===========================
def gen_report(concepts, diseases, retrieved):
    prompt=f"""
Generate chest X-ray impression (no headings). 
Conservative, factual, 4–6 sentences.

Diseases: {diseases}
Concept cues: {concepts}
Evidence: {" ".join(retrieved[:3])}

Rules:
- If "no acute disease", produce a normal study impression.
- Explicitly state heart, lungs, pleura, bones, final line.
- No hallucinations. No extra findings.
"""

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.1
    )
    return r.choices[0].message.content.strip()

# ===========================
# ✅ Run RAG + LLM
# ===========================
results=[]
for r in tqdm(test, desc="🩻 Running RAG pipeline"):
    gt = ((r.get("findings") or "") + " " + (r.get("impression") or "")).strip()
    concepts = extract_concepts(gt)
    diseases = extract_disease(gt)
    query = ", ".join(concepts+diseases)
    retrieved = rag(query, k=5)
    report = gen_report(concepts, diseases, retrieved)

    results.append({
        "uid": r["uid"],
        "image": r.get("image"),
        "ground_truth": gt,
        "predicted_disease": "; ".join(diseases),
        "concepts": "; ".join(concepts),
        "rag_retrieved": " || ".join(retrieved),
        "our_method": report
    })

df=pd.DataFrame(results)
df.to_csv("iu_rag_results_bioclincial.csv",index=False)
print("✅ Saved -> iu_rag_results_bioclincial.csv")
print(df.head())

# ===========================
# ✅ Metrics
# ===========================
print("\n📏 Computing metrics...")
import nltk, sacrebleu
nltk.download("punkt", quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction().method1
refs=df["ground_truth"].tolist(); gens=df["our_method"].tolist()
ref_tok=[nltk.word_tokenize(r.lower()) for r in refs]
gen_tok=[nltk.word_tokenize(g.lower()) for g in gens]

BLEU1=np.mean([sentence_bleu([ref_tok[i]],gen_tok[i],weights=(1,0,0,0),smoothing_function=smooth) for i in range(len(refs))])
BLEU2=np.mean([sentence_bleu([ref_tok[i]],gen_tok[i],weights=(.5,.5,0,0),smoothing_function=smooth) for i in range(len(refs))])
BLEU3=np.mean([sentence_bleu([ref_tok[i]],gen_tok[i],weights=(1/3,1/3,1/3,0),smoothing_function=smooth) for i in range(len(refs))])
BLEU4_sent=np.mean([sentence_bleu([ref_tok[i]],gen_tok[i],weights=(.25,.25,.25,.25),smoothing_function=smooth) for i in range(len(refs))])
BLEU4_corpus=sacrebleu.corpus_bleu(gens,[refs]).score

from rouge_score import rouge_scorer
scorer=rouge_scorer.RougeScorer(["rougeL"],use_stemmer=True)
ROUGE_L=np.mean([scorer.score(refs[i],gens[i])["rougeL"].fmeasure for i in range(len(refs))])

# CIDEr
try:
    from pycocoevalcap.cider.cider import Cider
except:
    subprocess.run([sys.executable,"-m","pip","install","-q","git+https://github.com/salaniz/pycocoevalcap"])
    from pycocoevalcap.cider.cider import Cider
cider=Cider()
gts={i:[refs[i]] for i in range(len(refs))}
res={i:[gens[i]] for i in range(len(gens))}
CIDEr_score,_=cider.compute_score(gts,res)

print("\n===== RESULTS =====")
print(f"BLEU-1: {BLEU1:.4f}")
print(f"BLEU-2: {BLEU2:.4f}")
print(f"BLEU-3: {BLEU3:.4f}")
print(f"BLEU-4: {BLEU4_sent:.4f}")
print(f"BLEU-4 Corpus: {BLEU4_corpus:.2f}")
print(f"ROUGE-L: {ROUGE_L:.4f}")
print(f"CIDEr: {CIDEr_score:.3f}")
print("===================\n✅ Done")
