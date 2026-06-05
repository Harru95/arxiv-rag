import os
import json
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag import load_index,hybrid_retrieve

load_dotenv()

# ── Test set ────────────────────────────────────────────────────────────────
# Each entry: question + keywords that MUST appear in retrieved chunks
# Keywords are lowercase, partial matches count
TEST_SET = [
    {
        "question": "What is scaled dot-product attention?",
        "relevant_keywords": ["query", "key", "value", "softmax", "scaling"]
    },
    {
        "question": "What is multi-head attention?",
        "relevant_keywords": ["multi-head", "parallel", "subspace", "concat"]
    },
    {
        "question": "What is the base model configuration?",
        "relevant_keywords": ["base", "model", "layers", "512", "heads"]
    },
    {
        "question": "What are the encoder and decoder components?",
        "relevant_keywords": ["encoder", "decoder", "stack", "layers"]
    },
    {
        "question": "What is positional encoding?",
        "relevant_keywords": ["positional", "encoding", "sine", "cosine", "position"]
    },
    {
        "question": "What datasets were used for training?",
        "relevant_keywords": ["wmt", "english", "german", "translation", "dataset"]
    },
    {
        "question": "What is the BLEU score achieved?",
        "relevant_keywords": ["bleu", "score", "28", "41"]
    },
    {
        "question": "What is feed-forward network in transformer?",
        "relevant_keywords": ["feed-forward", "ffn", "linear", "relu", "position-wise"]
    },
    {
        "question": "How does the model handle variable length sequences?",
        "relevant_keywords": ["attention", "sequence", "position", "mask"]
    },
    {
        "question": "What is dropout used for in the model?",
        "relevant_keywords": ["dropout", "regularization", "overfitting", "rate"]
    }
]

def chunk_is_relevant(chunk_text: str, keywords: list[str]) -> bool:
    """Check if a chunk contains at least 2 of the expected keywords."""
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches >= 2

def precision_at_k(retrieved_chunks, keywords: list[str], k: int) -> float:
    """Fraction of top-k chunks that are relevant."""
    top_k = retrieved_chunks[:k]
    relevant = sum(1 for chunk in top_k if chunk_is_relevant(chunk.page_content, keywords))
    return relevant / k

def reciprocal_rank(retrieved_chunks, keywords: list[str]) -> float:
    """1/rank of first relevant chunk. 0 if none found."""
    for i, chunk in enumerate(retrieved_chunks):
        if chunk_is_relevant(chunk.page_content, keywords):
            return 1.0 / (i + 1)
    return 0.0

def evaluate(k: int = 5):
    """Run evaluation on the test set."""
    
    print("Loading index...")
    index, chunks_list = load_index()
    if index is None:
        print("No index found. Run the app and index a paper first.")
        return
    
    precisions = []
    mrr_scores = []
    results = []
    
    print(f"\nRunning evaluation on {len(TEST_SET)} questions (k={k})...\n")
    print("-" * 60)
    
    for item in TEST_SET:
        question = item["question"]
        keywords = item["relevant_keywords"]
        
        chunks = hybrid_retrieve(index, chunks_list, question, k)
        
        p_at_k = precision_at_k(chunks, keywords, k)
        rr = reciprocal_rank(chunks, keywords)
        
        precisions.append(p_at_k)
        mrr_scores.append(rr)
        
        status = "✓" if p_at_k > 0 else "✗"
        print(f"{status} Q: {question[:50]}")
        print(f"  Precision@{k}: {p_at_k:.2f} | Reciprocal Rank: {rr:.2f}")
        print()
    
    mean_precision = sum(precisions) / len(precisions)
    mrr = sum(mrr_scores) / len(mrr_scores)
    
    print("-" * 60)
    print(f"RESULTS — Transformer Paper (Attention Is All You Need)")
    print(f"Mean Precision@{k}: {mean_precision:.3f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")
    print("-" * 60)
    
    # Save results to JSON
    output = {
        "paper": "arxiv.org/abs/1706.03762",
        "k": k,
        f"mean_precision_at_{k}": round(mean_precision, 3),
        "mrr": round(mrr, 3),
        "per_question": [
            {
                "question": TEST_SET[i]["question"],
                f"precision_at_{k}": round(precisions[i], 3),
                "reciprocal_rank": round(mrr_scores[i], 3)
            }
            for i in range(len(TEST_SET))
        ]
    }
    
    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to eval_results.json")
    return output

if __name__ == "__main__":
    evaluate(k=5)
