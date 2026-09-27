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
TEST_N = 50           
TOPK_TEXT = 40        
FINAL_K = 5           
ALPHA = 0.70          
BETA  = 0.20         
GAMMA = 0.10         

EMB_NPY = "iu_bioclinicalbert_embeddings.npy"
EMB_PT  = "iu_bioclinicalbert_dict.pt"

ALIGN_JSON = "alignment_concepts.json"   
GNN_WIDE = "gnn_priors_wide.csv"         

# ===========================
# 1) Key & client
# ===========================
api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openAPI_Key")
if not api_key:
    raise RuntimeError("❌ API key not found")

client = OpenAI(api_key=api_key)
print("🔑 OpenAI Key Loaded")

# ===========================
# 2) Dataset
# ===========================
ds = load_dataset("ykumards/open-i")
full = ds["train"]
test = full.select(range(TEST_N))
train = full.select(range(TEST_N, len(full)))
print(f"✅ Dataset Loaded | Train={len(train)} Test={len(test)}")

# ===========================
# 3) Load Embeddings + FAISS
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
print(f"✅ FAISS Ready {emb_train.shape}")

# ===========================
# 4) BioClinicalBERT Encoder
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

def embed_query(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        vec = outputs.last_hidden_state.mean(dim=1)
    return F.normalize(vec, p=2, dim=1).cpu().numpy()

# ===========================
# 5) Disease lexicon
# ===========================
chexpert = {
    "cardiomegaly":["cardiomegaly","heart enlargement"],
    "edema":["edema","pulmonary edema"],
    "consolidation":["consolidation","airspace disease"],
    "atelectasis":["atelectasis","collapse"],
    "effusion":["effusion","pleural effusion"],
    "pneumothorax":["pneumothorax"],
    "pneumonia":["pneumonia","infiltrate"],
    "pleural thickening":["pleural thickening"],
    "fibrosis":["fibrosis","interstitial"],
    "emphysema":["emphysema","hyperinflation"],
    "nodule/mass":["nodule","mass"],
    "lung opacity":["opacity","infiltrate"],
    "hernia":["hernia"],
    "support devices":["line","tube","pacemaker"]
}
neg_triggers=["no","without","absence of","free of","not seen","none"]

def is_neg(t,kw,window=50):
    pos = t.find(kw)
    if pos==-1:return False
    win = t[max(0,pos-window):pos]
    return any(n in win for n in neg_triggers)

def extract_disease(t):
    t=t.lower()
    f=[]
    for d,kws in chexpert.items():
        for kw in kws:
            if kw in t and not is_neg(t,kw):
                f.append(d);break
    return sorted(set(f)) or ["no acute disease"]

# ===========================
# 6) Heuristic concepts (fallback)
# ===========================
def extract_concepts_heur(t):
    t=t.lower()
    c=[]
    if "heart size" in t or "cardiac silhouette" in t: c.append("heart size normal")
    if "clear lungs" in t or "lungs are clear" in t: c.append("clear lungs")
    if "no pneumothorax" in t: c.append("no pneumothorax")
    if "no effusion" in t: c.append("no effusion")
    return sorted(set(c)) or ["no acute abnormality"]

# ===========================
# 7) Load Alignment JSON
# ===========================
if os.path.exists(ALIGN_JSON):
    with open(ALIGN_JSON,"r") as f: align_map=json.load(f)
    print(f"🧷 Alignment Loaded: {len(align_map)} samples")
else:
    align_map={}
    print("ℹ️ No alignment JSON found — using heuristics")

# ===========================
# 8) Load GNN Priors
# ===========================
if os.path.exists(GNN_WIDE):
    gnn_df=pd.read_csv(GNN_WIDE)
    disease_cols=[c for c in gnn_df.columns if c not in ["uid","split"]]
    gnn_dict={str(r.uid):r[disease_cols].values.astype("float32") for _,r in gnn_df.iterrows()}
    print(f"🧠 Loaded GNN Priors: {len(gnn_dict)}")
else:
    gnn_dict={}
    print("ℹ️ No GNN priors — Gamma will act weak")

def gnn_similarity(uid1,uid2):
    v1=gnn_dict.get(str(uid1)); v2=gnn_dict.get(str(uid2))
    if v1 is None or v2 is None: return 0.0
    return float(np.dot(v1,v2))

# ===========================
# 9) KG Co-occurrence Graph
# ===========================
docs_tokens=[set(extract_disease(t)) for t in corpus]
from collections import Counter
df_counts=Counter()
pair_counts=Counter()
for toks in docs_tokens:
    for a in toks: df_counts[a]+=1
    toks=sorted(toks)
    for i in range(len(toks)):
        for j in range(i+1,len(toks)):
            pair_counts[(toks[i],toks[j])]+=1

N=len(docs_tokens)
def pmi(a,b):
    pa=df_counts[a]/N; pb=df_counts[b]/N; pab=pair_counts[(a,b)]/N
    return max(math.log((pab/(pa*pb))+1e-9),0)

neighbor_score={a:{b:pmi(a,b) for b in df_counts if (a,b) in pair_counts} for a in df_counts}
term_in_doc=docs_tokens

def kg_score(query,doc):
    return sum(neighbor_score.get(q,{}).get(d,0) for q in query for d in doc)

# ===========================
# 10) RAG retrieval
# ===========================
def graph_rag(query_terms,uid):
    q=embed_query(", ".join(query_terms)); faiss.normalize_L2(q)
    D,I=index.search(q,TOPK_TEXT)
    text=(D[0]-D[0].min())/(np.ptp(D[0])+1e-9)
    kg=np.array([kg_score(query_terms, term_in_doc[i]) for i in I[0]])
    kg=(kg-kg.min())/(np.ptp(kg)+1e-9)
    gnn=np.array([gnn_similarity(uid,train_uids[i]) for i in I[0]])
    gnn=(gnn-gnn.min())/(np.ptp(gnn)+1e-9)
    fused=ALPHA*text+BETA*kg+GAMMA*gnn
    order=np.argsort(-fused)[:FINAL_K]
    return [corpus[i][:300] for i in order],{"text":text[order],"kg":kg[order],"gnn":gnn[order],"fused":fused[order]}

# ===========================
# 11) Generate Impression
# ===========================
def prompt(concepts,diseases,retrieved,score):
    expl="\n".join([f"- Doc{i+1}: fused={score['fused'][i]:.2f}, text={score['text'][i]:.2f}, kg={score['kg'][i]:.2f}, gnn={score['gnn'][i]:.2f}" for i in range(len(retrieved))])
    return f"""
You are a radiologist. Write the IMPRESSION only.
Detected: {diseases}
Concepts: {concepts}

Cases:
{chr(10).join("- "+r for r in retrieved)}

Evidence:
{expl}

Be factual, mention heart size, lungs, pleura, and bones. Avoid hallucinations.
"""

def gen(con,dz,ret,sc):
    r=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt(con,dz,ret,sc)}],
        temperature=0.1,max_tokens=260
    )
    return r.choices[0].message.content.strip()

# ===========================
# 12) Run
# ===========================
results=[]
for row in tqdm(test,desc="Running Full Pipeline"):
    uid=str(row["uid"])
    gt=((row.get("findings") or "")+" "+(row.get("impression") or "")).strip()

    concepts = align_map.get(uid) if uid in align_map else extract_concepts_heur(gt)
    diseases = extract_disease(gt)
    query = sorted(set(concepts+diseases))

    retrieved, scores = graph_rag(query,uid)
    report = gen(concepts,diseases,retrieved,scores)

    results.append({
        "uid":uid,
        "ground_truth":gt,
        "concepts":"; ".join(concepts),
        "diseases":"; ".join(diseases),
        "retrieved":" || ".join(retrieved),
        "our_output":report
    })

df=pd.DataFrame(results)
out="iu_rag_alignment_final.csv"
df.to_csv(out,index=False)
print(f"✅ Saved -> {out}")
print(df.head(3))

# ===========================
# 13) Metrics
# ===========================
print("\n📏 Metrics...")
import nltk, sacrebleu
nltk.download("punkt",quiet=True)
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
smooth=SmoothingFunction().method1
refs=df["ground_truth"]; gens=df["our_output"]

ref_tok=[nltk.word_tokenize(r.lower()) for r in refs]
gen_tok=[nltk.word_tokenize(g.lower()) for g in gens]

BLEU1=np.mean([sentence_bleu([ref_tok[i]],gen_tok[i],weights=(1,0,0,0),smoothing_function=smooth) for i in range(len(refs))])
BLEU2=np.mean([sentence_bleu([ref_tok[i]],gen_tok[i],weights=(.5,.5,0,0),smoothing_function=smooth) for i in range(len(refs))])
BLEU3=np.mean([sentence_bleu([ref_tok[i]],gen_tok[i],weights=(1/3,1/3,1/3,0),smoothing_function=smooth) for i in range(len(refs))])
BLEU4=np.mean([sentence_bleu([ref_tok[i]],gen_tok[i],weights=(.25,.25,.25,.25),smoothing_function=smooth) for i in range(len(refs))])
BLEU_c=sacrebleu.corpus_bleu(gens.tolist(),[refs.tolist()]).score

from rouge_score import rouge_scorer
sc=rouge_scorer.RougeScorer(["rougeL"],use_stemmer=True)
ROUGE=np.mean([sc.score(refs[i],gens[i])["rougeL"].fmeasure for i in range(len(refs))])

try:
    from pycocoevalcap.cider.cider import Cider
except:
    subprocess.run([sys.executable,"-m","pip","install","-q","git+https://github.com/salaniz/pycocoevalcap"])
    from pycocoevalcap.cider.cider import Cider

cider=Cider(); gts={i:[refs[i]] for i in range(len(refs))}; res={i:[gens[i]] for i in range(len(gens))}
CIDEr,_=cider.compute_score(gts,res)

print(f"\n===== FINAL PIPELINE RESULTS (WITH ALIGNMENT) =====")
print(f"BLEU-1: {BLEU1:.4f}")
print(f"BLEU-2: {BLEU2:.4f}")
print(f"BLEU-3: {BLEU3:.4f}")
print(f"BLEU-4: {BLEU4:.4f}")
print(f"BLEU-4 Corpus: {BLEU_c:.2f}")
print(f"ROUGE-L: {ROUGE:.4f}")
print(f"CIDEr: {CIDEr:.3f}")
print("=================================================")
print(f"📄 File: {out}")
