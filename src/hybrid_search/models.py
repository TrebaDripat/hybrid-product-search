import json
import os
import threading
from functools import cached_property
from importlib.resources import files

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from hybrid_search.domain import fingerprint

SPECS = json.loads(files("hybrid_search").joinpath("models.json").read_text())


def normalized(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms < 1e-12).any():
        raise ValueError("model returned invalid or zero-length vectors")
    return vectors / norms


class OnnxModels:
    """Pinned local inference only: failures never fall back to synthetic scores."""

    signature = fingerprint(SPECS["embedding"])

    def __init__(self, threads: int = 2):
        if not 1 <= threads <= 16:
            raise ValueError("inference threads must be between 1 and 16")
        self.threads = threads
        # Fast tokenizers change padding/truncation settings during calls. Guard
        # loading and inference together to bound CPU/memory and avoid races.
        self.lock = threading.RLock()

    def _load(self, kind):
        spec = SPECS[kind]
        options = dict(repo_id=spec["repository"], revision=spec["revision"])
        config = None
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            config = hf_hub_download(filename=name, **options)
        model = hf_hub_download(filename=spec["file"], **options)
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.dirname(config),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = self.threads
        settings.inter_op_num_threads = 1
        session = ort.InferenceSession(
            model, sess_options=settings, providers=["CPUExecutionProvider"]
        )
        return tokenizer, session

    @cached_property
    def encoder(self):
        return self._load("embedding")

    @cached_property
    def reranker(self):
        return self._load("reranker")

    @staticmethod
    def _run(session, tokens):
        feed = {}
        for item in session.get_inputs():
            if item.name == "token_type_ids" and item.name not in tokens:
                feed[item.name] = np.zeros_like(tokens["input_ids"], dtype=np.int64)
            else:
                feed[item.name] = np.asarray(tokens[item.name], dtype=np.int64)
        return session.run(None, feed)[0]

    def _encode(self, texts, prefix):
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        with self.lock:
            tokenizer, session = self.encoder
            vectors = []
            for offset in range(0, len(texts), 8):
                tokens = tokenizer(
                    [prefix + text for text in texts[offset : offset + 8]],
                    padding=True,
                    truncation=True,
                    max_length=SPECS["embedding"]["max_tokens"],
                    return_tensors="np",
                )
                hidden = self._run(session, tokens)
                if hidden.ndim != 3 or hidden.shape[2] != 384:
                    raise ValueError("unexpected E5 token embedding shape")
                mask = np.asarray(tokens["attention_mask"], dtype=np.float32)[..., None]
                vectors.append(normalized((hidden * mask).sum(axis=1) / mask.sum(axis=1)))
            return np.concatenate(vectors)

    def passages(self, texts):
        return self._encode(texts, SPECS["embedding"]["passage_prefix"])

    def query(self, text):
        return self._encode([text], SPECS["embedding"]["query_prefix"])[0]

    def rerank(self, query, documents):
        if not documents:
            return []
        with self.lock:
            tokenizer, session = self.reranker
            scores = []
            for offset in range(0, len(documents), 8):
                batch = documents[offset : offset + 8]
                tokens = tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=SPECS["reranker"]["max_tokens"],
                    return_tensors="np",
                )
                logits = self._run(session, tokens)
                if logits.shape != (len(batch), 1) or not np.isfinite(logits).all():
                    raise ValueError("unexpected BGE logit shape or value")
                scores.extend(float(value) for value in logits[:, 0])
            return scores

    def warmup(self):
        self.query("portable storage")
        self.rerank("portable storage", ["A portable solid-state drive."])
