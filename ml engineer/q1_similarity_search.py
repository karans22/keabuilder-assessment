"""
q1_similarity_search.py

Small similarity search engine for KeaBuilder.

Use case: when a new lead comes in, find existing leads/prompts that look similar
so we can reuse a response or flag duplicates. Same logic could work for matching
user-typed prompts to pre-built funnel templates.

I kept this dependency-free on purpose -- no sklearn, no torch, no nothing.
TF-IDF + cosine similarity is plenty for this scale and it runs anywhere.
If we needed semantic (meaning-level) matching, I'd swap in sentence-transformers
and pgvector, but that's overkill for the demo.
"""

import math
from collections import Counter
from typing import List


def tokenize(text: str) -> List[str]:
    # lowercase + strip punctuation, nothing fancy
    return text.lower().replace(",", " ").replace(".", " ").replace("?", " ").split()


def build_tfidf(corpus: List[str]):
    """
    Compute TF-IDF for a list of strings.
    Returns the vectors and the shared vocabulary.
    Nothing clever here -- just the standard formula with +1 smoothing.
    """
    tokenized = [tokenize(doc) for doc in corpus]
    vocab = sorted(set(w for doc in tokenized for w in doc))
    N = len(corpus)

    idf = {
        word: math.log((N + 1) / (sum(1 for doc in tokenized if word in doc) + 1)) + 1
        for word in vocab
    }

    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        vectors.append({
            word: (tf[word] / len(tokens)) * idf[word]
            for word in vocab
        })

    return vectors, vocab, idf


def cosine(a: dict, b: dict) -> float:
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in a)
    mag_a = math.sqrt(sum(v**2 for v in a.values()))
    mag_b = math.sqrt(sum(v**2 for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class LeadMatcher:
    """
    Index a bunch of existing lead texts, then search for the closest
    match to any new query. That's it.

    Usage:
        matcher = LeadMatcher()
        matcher.index(documents)   # list of {id, text, metadata}
        results = matcher.search("automate my email follow-ups", top_k=2)
    """

    def __init__(self):
        self.docs = []
        self.vectors = []
        self.vocab = []
        self.idf = {}

    def index(self, documents: List[dict]):
        self.docs = documents
        corpus = [d["text"] for d in documents]
        self.vectors, self.vocab, self.idf = build_tfidf(corpus)

    def _vectorize(self, text: str) -> dict:
        # turn a query string into a TF-IDF vector using the same vocab + idf
        tokens = tokenize(text)
        if not tokens:
            return {w: 0.0 for w in self.vocab}
        tf = Counter(tokens)
        return {
            word: (tf.get(word, 0) / len(tokens)) * self.idf.get(word, 0)
            for word in self.vocab
        }

    def search(self, query: str, top_k: int = 1) -> List[dict]:
        if not self.docs:
            raise RuntimeError("Nothing indexed yet. Call index() first.")

        q_vec = self._vectorize(query)
        scored = sorted(
            [(cosine(q_vec, vec), i) for i, vec in enumerate(self.vectors)],
            reverse=True
        )

        return [
            {
                "rank": rank + 1,
                "id": self.docs[idx]["id"],
                "text": self.docs[idx]["text"],
                "score": round(score, 4),
                **self.docs[idx].get("metadata", {})
            }
            for rank, (score, idx) in enumerate(scored[:top_k])
        ]


# ─── DEMO ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # pretend these are leads already in the system
    existing_leads = [
        {
            "id": "lead_001",
            "text": "We need to automate lead capture and follow-up emails for our sales team",
            "metadata": {"classification": "HOT", "source": "funnel_form"}
        },
        {
            "id": "lead_002",
            "text": "Looking for a chatbot to handle incoming queries on our website",
            "metadata": {"classification": "WARM", "source": "landing_page"}
        },
        {
            "id": "lead_003",
            "text": "We want to build funnels and track conversion rates",
            "metadata": {"classification": "WARM", "source": "pricing_page"}
        },
        {
            "id": "lead_004",
            "text": "Just exploring tools for social media scheduling",
            "metadata": {"classification": "COLD", "source": "blog"}
        },
        {
            "id": "lead_005",
            "text": "Need CRM integration and email drip campaigns for our B2B sales process",
            "metadata": {"classification": "HOT", "source": "funnel_form"}
        },
    ]

    matcher = LeadMatcher()
    matcher.index(existing_leads)

    queries = [
        "automate follow-up emails and lead management",
        "chatbot for customer support on website",
        "build sales funnels and measure performance",
        "drip campaign for enterprise B2B clients",
    ]

    print("KeaBuilder Lead Matcher — demo\n")
    for q in queries:
        results = matcher.search(q, top_k=2)
        print(f"Query: '{q}'")
        for r in results:
            print(f"  #{r['rank']} [{r['score']}] {r['id']} ({r.get('classification')}) — {r['text'][:60]}...")
        print()
