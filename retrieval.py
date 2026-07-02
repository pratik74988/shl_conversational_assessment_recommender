"""
Hybrid retrieval: BM25 (exact/keyword) + dense embeddings (semantic).
Combined via Reciprocal Rank Fusion (RRF).
BM25 catches exact test names like "OPQ32r", "Java 8 (New)".
Dense catches semantic matches like "communication skills" → OPQ.
"""

import json
import re
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# Maps catalog key strings to single-letter test_type codes used in API response
KEY_TO_CODE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Assessment Exercises": "E",
    "Development & 360": "D",
}

# Seniority terms → job level strings in catalog
SENIORITY_MAP = {
    "entry": ["Entry-Level", "Graduate"],
    "mid": ["Mid-Professional", "Professional Individual Contributor"],
    "senior": ["Professional Individual Contributor", "Mid-Professional"],
    "manager": ["Manager", "Front Line Manager", "Supervisor"],
    "director": ["Director"],
    "executive": ["Executive", "Director"],
    "graduate": ["Graduate", "Entry-Level"],
}


class CatalogRetriever:
    def __init__(self, catalog_path: str = "catalog.json"):
        self.catalog = self._load_catalog(catalog_path)
        self._url_set = {item["link"] for item in self.catalog}
        self._name_index = {item["name"].lower(): item for item in self.catalog}
        self._build_indexes()

    # ------------------------------------------------------------------ #
    #  Loading & indexing                                                  #
    # ------------------------------------------------------------------ #

    def _load_catalog(self, path: str) -> list:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = [item for item in data if item.get("status") == "ok"]
        print(f"[retrieval] Loaded {len(items)} catalog items.")
        return items

    def _make_doc(self, item: dict) -> str:
        """Concatenate all searchable fields into one string."""
        parts = [
            item.get("name", ""),
            item.get("description", ""),
            " ".join(item.get("keys", [])),
            " ".join(item.get("job_levels", [])),
        ]
        return " ".join(p for p in parts if p)

    def _tokenize(self, text: str) -> list:
        return re.findall(r"\w+", text.lower())

    def _build_indexes(self):
        self._docs = [self._make_doc(item) for item in self.catalog]
        tokenized = [self._tokenize(doc) for doc in self._docs]
        self.bm25 = BM25Okapi(tokenized)

        print("[retrieval] Loading embedding model (all-MiniLM-L6-v2)...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._embeddings = self._model.encode(
            self._docs, batch_size=64, show_progress_bar=False
        )
        print("[retrieval] Index ready.")

    # ------------------------------------------------------------------ #
    #  Core search                                                         #
    # ------------------------------------------------------------------ #

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0

    def _rrf(self, *rank_lists: list, k: int = 60) -> list:
        """Reciprocal Rank Fusion across any number of ranked lists."""
        scores: dict[int, float] = {}
        for ranks in rank_lists:
            for position, idx in enumerate(ranks):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + position + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list:
        """
        Hybrid search with optional post-filters.
        filters keys: test_types_wanted, test_types_excluded, remote, seniority
        """
        # BM25 ranking
        tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_ranks = np.argsort(bm25_scores)[::-1].tolist()

        # Dense ranking
        qvec = self._model.encode([query])[0]
        cosine_scores = [self._cosine_sim(qvec, e) for e in self._embeddings]
        dense_ranks = np.argsort(cosine_scores)[::-1].tolist()

        # Fuse
        fused = self._rrf(bm25_ranks, dense_ranks)

        filters = filters or {}
        wanted_types = set(filters.get("test_types_wanted") or [])
        excluded_types = set(filters.get("test_types_excluded") or [])
        remote_only = filters.get("remote")
        seniority = filters.get("seniority")
        wanted_levels = set(SENIORITY_MAP.get(seniority, [])) if seniority else set()

        results = []
        for idx in fused:
            item = self.catalog[idx]
            item_keys = set(item.get("keys", []))

            if wanted_types and not item_keys.intersection(wanted_types):
                continue
            if excluded_types and item_keys.intersection(excluded_types):
                continue
            if remote_only is True and item.get("remote") != "yes":
                continue
            if wanted_levels:
                item_levels = set(item.get("job_levels", []))
                if not item_levels.intersection(wanted_levels):
                    continue

            results.append(item)
            if len(results) >= top_k:
                break

        # If filters eliminated everything, fall back to unfiltered top-k
        if not results:
            results = [self.catalog[i] for i in fused[:top_k]]

        return results

    # ------------------------------------------------------------------ #
    #  Lookup helpers                                                      #
    # ------------------------------------------------------------------ #

    def get_by_name(self, name: str) -> dict | None:
        """Exact then substring match by name."""
        key = name.lower().strip()
        if key in self._name_index:
            return self._name_index[key]
        for item_name_lower, item in self._name_index.items():
            if key in item_name_lower or item_name_lower in key:
                return item
        return None

    def validate_url(self, url: str) -> bool:
        return url in self._url_set

    # ------------------------------------------------------------------ #
    #  Formatting                                                          #
    # ------------------------------------------------------------------ #

    def format_for_context(self, item: dict) -> str:
        """Human-readable block for LLM context window."""
        return (
            f"Name: {item['name']}\n"
            f"URL: {item['link']}\n"
            f"Type: {', '.join(item.get('keys', []))}\n"
            f"Job Levels: {', '.join(item.get('job_levels', []))}\n"
            f"Duration: {item.get('duration') or 'Not specified'}\n"
            f"Remote: {item.get('remote', 'unknown')} | "
            f"Adaptive: {item.get('adaptive', 'unknown')}\n"
            f"Description: {item.get('description', '')}\n"
            "---"
        )

    def to_recommendation(self, item: dict) -> dict:
        keys = item.get("keys", [])
        test_type = KEY_TO_CODE.get(keys[0], "K") if keys else "K"
        return {
            "name": item["name"],
            "url": item["link"],
            "test_type": test_type,
        }
