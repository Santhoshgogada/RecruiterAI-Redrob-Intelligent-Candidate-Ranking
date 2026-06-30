"""
jd.py — Job Description constants for Senior AI Engineer (Founding Team) @ Redrob AI

All skill/keyword lists below were calibrated against a 2,000-candidate random
sample of the real 100,000-row candidates.jsonl — not guessed. Specifically:
  - AI skill vocabulary = exact skill names that appear in the dataset
    (e.g. "RAG", "Fine-tuning LLMs", "QLoRA" are literal skill.name values)
  - Disqualifier titles = the most common non-AI titles found in the data
  - Work-description AI signal terms = calibrated so that real ML-titled
    candidates average ~4 hits in their career_history descriptions, while
    honeypot candidates (AI skills listed, unrelated title) average ~1 or 0.
"""

JD_TEXT = """
Senior AI Engineer — Founding Team
Redrob AI | Series A | Noida / Pune

You'll be one of the first AI engineers at Redrob AI, building the
intelligence layer of India's AI Operating System. 0→1 role: you're
architecting systems, not maintaining them.

What you'll own:
- Design and ship LLM-powered features end-to-end: RAG, retrieval, ranking,
  generation, evaluation
- Fine-tune and align models (SFT, RLHF, DPO, LoRA/QLoRA)
- Build and own inference infrastructure — GPU fleet, serving latency,
  throughput, cost
- Create evaluation frameworks: offline evals, human preference data,
  red-teaming

What you need:
- 3–8 years of hands-on ML/AI engineering (engineering, not pure research
  or data science)
- Deep LLM experience: fine-tuning, RAG, prompt engineering, evaluation
- Strong Python; PyTorch or JAX
- Production experience — shipped models real users depend on
- Vector databases, embeddings, retrieval systems

The right answer is NOT to find candidates whose skills section contains
the most AI keywords. A candidate who lists every AI keyword but whose
actual title and work history show no AI engineering substance is not a fit.
"""

# ── Exact AI skill vocabulary found in the dataset (skill.name values) ──────
# Counted in a 2,000-candidate sample. These ARE the skills that appear —
# not a generic guess.
MUST_HAVE_SKILLS = {
    "llms", "fine-tuning llms", "rag", "hugging face transformers",
    "sentence transformers", "prompt engineering", "vector search",
    "embeddings", "semantic search", "pytorch", "tensorflow",
    "machine learning", "deep learning", "nlp", "qlora", "lora",
    "pgvector", "faiss", "pinecone", "milvus", "weaviate",
}

NICE_TO_HAVE_SKILLS = {
    "computer vision", "image classification", "speech recognition", "tts",
    "gans", "reinforcement learning", "statistical modeling",
    "weights & biases", "bentoml", "github actions", "mlflow",
    "kubernetes", "docker", "aws", "gcp", "azure", "cuda", "triton",
    "vllm", "apache beam",
}

# ── EXHAUSTIVE title list ──────────────────────────────────────────────────
# The dataset uses a CLOSED vocabulary of exactly 34 distinct current_title
# values (confirmed via direct enumeration over a 2,000-candidate sample —
# no new titles appeared beyond this list, so this is treated as exhaustive
# rather than a fuzzy pattern guess).
#
# Titles are bucketed into 4 tiers by genuine AI/ML engineering relevance:

# Tier A — core AI/ML engineering roles. Exactly what the JD wants.
STRONG_TITLE_PATTERNS = [
    "ml engineer", "junior ml engineer", "applied ml engineer",
    "ai research engineer", "ai specialist",
    "senior software engineer (ml)",
]

# Tier B — adjacent technical roles. Could plausibly do the job with some
# ramp-up; partial credit, not full credit.
ADJACENT_TITLE_PATTERNS = [
    "data scientist", "data engineer", "senior data engineer",
    "analytics engineer", "backend engineer", "senior software engineer",
    "software engineer", "cloud engineer", "devops engineer",
]

# Tier C — technical but not ML-adjacent (web/mobile dev, QA). Weak signal,
# not disqualified, just low relevance.
WEAK_TECHNICAL_TITLES = [
    ".net developer", "java developer", "full stack developer",
    "frontend engineer", "mobile developer", "data analyst",
    "qa engineer",
]

# Tier D — the dataset's intentional non-technical noise population
# (confirmed exhaustively: these 12 titles account for ~70% of the dataset
# by design, per real-sample frequency counts).
DISQUALIFIER_TITLE_PATTERNS = [
    "marketing manager", "mechanical engineer", "customer support",
    "content writer", "civil engineer", "sales executive",
    "business analyst", "graphic designer", "accountant", "hr manager",
    "operations manager", "project manager",
]

# Headline phrases that are a strong honeypot tell in this dataset —
# candidates explicitly self-describe as exploring/dabbling, not doing.
HONEYPOT_HEADLINE_PHRASES = [
    "exploring ai", "exploring genai", "ai enthusiast", "genai explorer",
    "ai & genai applications", "generative ai explorer", "ai curious",
    "learning ai", "interested in ai",
]

# Career-description AI signal terms. Calibrated against real data:
# legitimate ML-titled candidates average ~4 hits per profile;
# honeypots average ~1 or 0.
WORK_DESCRIPTION_KEYWORDS = [
    "llm", "rag", "fine-tun", "pytorch", "tensorflow", "transformer",
    "embedding", "vector", "rlhf", "dpo", "lora", "model serving",
    "inference", "deployed model", "trained model", "ml pipeline",
    "feature engineering", "model deployment", "production model",
    "neural network", "reinforcement learning", "prompt",
]

# Companies/institutions known for strong AI/ML signal — kept intentionally
# broad since the dataset uses placeholder/fictional company names
# (Mindtree, Dunder Mifflin, Globex, Wysa, Haptik etc.) alongside real
# tech names. Real tech names get the bonus; fictional ones are neutral.
STRONG_AI_COMPANIES = {
    "google", "deepmind", "openai", "anthropic", "meta", "microsoft",
    "amazon", "apple", "nvidia", "hugging face", "cohere",
    "wysa", "haptik", "ola", "flipkart", "zomato", "swiggy", "meesho",
    "razorpay", "cred", "groww", "zepto", "sarvam", "krutrim",
}

# Education tier_1 = top institutions per the dataset's own tiering system
TOP_EDUCATION_TIER = "tier_1"
