import os, sys, json, math, warnings
import numpy as np, pandas as pd
from tqdm import tqdm
import torch, faiss
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from collections import Counter, defaultdict

warnings.filterwarnings("ignore", category=UserWarning)

# ===========================
# CONFIG
# ===========================
BASE_DIR = "/content/drive/MyDrive/Iqra Jannat/New_code/BiomedCLIP_openclip/IU_pipeline_runs"
os.makedirs(BASE_DIR, exist_ok=True)
os.chdir(BASE_DIR)

TEST_N = 50
TOPK_TEXT = 40
FINAL_K = 5
ALPHA, BETA, GAMMA = 0.70, 0.20, 0.10

EMB_PT = "iu_bioclinicalbert_dict.pt"
EMB_NPY = "iu_bioclinicalbert_embeddings.npy"
ALIGN_JSON = "alignment_concepts.json"
GNN_WIDE = "gnn_priors_wide.csv"

# ===========================
# OPENAI KEY
# ===========================
api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openAPI_Key")
if not api_key:
    raise RuntimeError("❌ Set your OpenAI API key first")
client = OpenAI(api_key=api_key)
print("🔑 OpenAI Key Loaded")

# ===========================
# DATASET
# ===========================
ds = load_dataset("ykumards/open-i")
full = ds["train"]
test = full.select(range(TEST_N))
train = full.select(range(TEST_N, len(full)))
print(f"✅ Dataset Loaded | Train={len(train)} Test={len(test)}")

# ===========================
# EMBEDDINGS + FAISS
# ===========================
print("📥 Loading Embeddings...")
uid_to_vec = torch.load(EMB_PT, weights_only=False)
emb_all = np.load(EMB_NPY)
train_uids, corpus = [], []
for r in train:
    txt = ((r.get("findings") or "") + " " + (r.get("impression") or "")).strip()
    if (r["uid"] in uid_to_vec) and txt:
        train_uids.append(r["uid"])
        corpus.append(txt)

emb_train = np.vstack([uid_to_vec[u] for u in train_uids]).astype("float32")
faiss.normalize_L2(emb_train)
index = faiss.IndexFlatIP(emb_train.shape[1])
index.add(emb_train)
print(f"✅ FAISS Index Ready: {emb_train.shape}")

# ===========================
# ENCODER (BioClinicalBERT)
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

def embed_query(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        vec = model(**inputs).last_hidden_state.mean(dim=1)
    return F.normalize(vec, p=2, dim=1).cpu().numpy()

# ===========================
# CHEXPERT DISEASE LIST
# ===========================
chexpert = {
    "cardiomegaly":["cardiomegaly","heart enlargement"],
    "edema":["edema","pulmonary edema"],
    "effusion":["effusion","pleural effusion"],
    "pneumothorax":["pneumothorax"],
    "pneumonia":["pneumonia","infiltrate"],
    "atelectasis":["atelectasis","collapse"],
    "fibrosis":["fibrosis"],
    "emphysema":["emphysema"],
    "consolidation":["consolidation"],
}
neg_triggers = ["no","without","absence of","free of","not seen","none"]

def is_neg(t,kw,window=50):
    pos = t.find(kw)
    if pos==-1: return False
    win = t[max(0,pos-window):pos]
    return any(n in win for n in neg_triggers)

def extract_disease(t):
    t=t.lower(); found=[]
    for d,kws in chexpert.items():
        for kw in kws:
            if kw in t and not is_neg(t,kw): found.append(d); break
    return sorted(set(found)) or ["no acute disease"]

# ===========================
# LOAD ALIGNMENT & GNN PRIORS
# ===========================
with open(ALIGN_JSON,"r") as f: align_map=json.load(f)
gnn_df=pd.read_csv(GNN_WIDE)
disease_cols=[c for c in gnn_df.columns if c not in ["uid","split"]]
gnn_dict={str(r.uid):r[disease_cols].values.astype("float32") for _,r in gnn_df.iterrows()}
print(f"🧠 GNN Priors: {len(gnn_dict)} | Alignment: {len(align_map)}")

# Clean alignment concepts
for k,v in align_map.items():
    if isinstance(v,list) and len(v)>0:
        align_map[k]=[x[0] if isinstance(x,(list,tuple)) else x for x in v]

# ===========================
# LIGHT KG (CO-OCCURRENCE)
# ===========================
docs_tokens=[]
for t in corpus:
    toks=set(extract_disease(t))
    if "heart size" in t: toks.add("heart size")
    if "clear lungs" in t: toks.add("clear lungs")
    if "no effusion" in t: toks.add("no effusion")
    docs_tokens.append(toks)

df_counts=Counter(); pair_counts=Counter()
for toks in docs_tokens:
    for a in toks: df_counts[a]+=1
    toks=sorted(toks)
    for i in range(len(toks)):
        for j in range(i+1,len(toks)): pair_counts[(toks[i],toks[j])]+=1
N=len(docs_tokens)
def pmi(a,b):
    pa=df_counts[a]/N; pb=df_counts[b]/N; pab=pair_counts[(a,b)]/N
    return max(math.log((pab/(pa*pb))+1e-9),0)
neighbor_score=defaultdict(dict)
for (a,b) in pair_counts:
    w=pmi(a,b)
    if w>0: neighbor_score[a][b]=w; neighbor_score[b][a]=w
term_in_doc=docs_tokens

# ===========================
# REASONING HELPERS
# ===========================
def top_gnn_diseases(uid,k=3):
    v=gnn_dict.get(str(uid))
    if v is None: return []
    arr=np.array(v); idx=np.argsort(-arr)[:k]
    return [disease_cols[i].lower() for i in idx]

def build_reason_paths(concepts,k_per_concept=1):
    paths=[]; dset=set(chexpert.keys())
    for c in concepts:
        neigh=neighbor_score.get(c,{})
        pairs=[(d,w) for d,w in neigh.items() if d in dset]
        pairs.sort(key=lambda x:-x[1])
        for d,w in pairs[:k_per_concept]:
            paths.append(f"{c} → {d} (PMI {w:.2f})")
    return paths[:4]

# ===========================
# GRAPH-AWARE RETRIEVAL
# ===========================
def kg_score(query,doc):
    return sum(neighbor_score.get(q,{}).get(d,0) for q in query for d in doc)

def graph_rag(query,uid):
    q=embed_query(", ".join(query)); faiss.normalize_L2(q)
    D,I=index.search(q,TOPK_TEXT)
    text=(D[0]-D[0].min())/(np.ptp(D[0])+1e-9)
    kg=np.array([kg_score(query,term_in_doc[i]) for i in I[0]])
    kg=(kg-kg.min())/(np.ptp(kg)+1e-9)
    gnn=np.array([float(np.dot(gnn_dict.get(str(uid),np.zeros(len(disease_cols))),
                               gnn_dict.get(str(train_uids[i]),np.zeros(len(disease_cols))))) for i in I[0]])
    gnn=(gnn-gnn.min())/(np.ptp(gnn)+1e-9)
    fused=ALPHA*text+BETA*kg+GAMMA*gnn
    order=np.argsort(-fused)[:FINAL_K]
    return [corpus[i][:300] for i in order],{"text":text[order],"kg":kg[order],"gnn":gnn[order],"fused":fused[order]}

# ===========================
# PROMPT + GENERATION
# ===========================
def prompt(con,dz,ret,sc,paths):
    rp="\n".join(["- "+p for p in paths]) if paths else "- (no biomarker→disease link)"
    expl="\n".join([f"- Doc{i+1}: fused={sc['fused'][i]:.2f}, text={sc['text'][i]:.2f}, kg={sc['kg'][i]:.2f}, gnn={sc['gnn'][i]:.2f}" for i in range(len(ret))])
    return f"""
You are a radiologist. Generate the IMPRESSION only.

Detected diseases: {dz}
Structural concepts: {con}

Reasoning chains (biomarker → disease):
{rp}

Retrieved prior cases:
{chr(10).join("- "+r for r in ret)}

Evidence weights:
{expl}

Be concise (4–6 sentences), factual, and avoid hallucinations.
"""

def gen(c,d,r,s,p):
    r_=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt(c,d,r,s,p)}],
        temperature=0.1,max_tokens=260
    )
    return r_.choices[0].message.content.strip()

# ===========================
# MAIN LOOP
# ===========================
results=[]
for row in tqdm(test,desc="🧭 Full RAG Pipeline (Reasoning Path)"):
    uid=str(row["uid"])
    gt=((row.get("findings") or "")+" "+(row.get("impression") or "")).strip()
    concepts=align_map.get(uid,[])
    diseases=top_gnn_diseases(uid) or extract_disease(gt)
    query=sorted(set(concepts+diseases))
    ret,sc=graph_rag(query,uid)
    paths=build_reason_paths(concepts)
    report=gen(concepts,diseases,ret,sc,paths)
    results.append({
        "uid":uid,"concepts":"; ".join(concepts),"diseases":"; ".join(diseases),
        "reason_paths":" || ".join(paths),"retrieved":" || ".join(ret),
        "ground_truth":gt,"our_output":report
    })

df=pd.DataFrame(results)
out="iu_rag_reasoning_results.csv"
df.to_csv(out,index=False)
print(f"💾 Saved -> {out}")
