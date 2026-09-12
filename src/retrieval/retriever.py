import threading

import torch
import torch.nn.functional as F
import faiss
import json
import numpy as np
from transformers import CLIPImageProcessor, CLIPTokenizer, AutoModel
from PIL import Image


class Retriever:
    """Visual retriever based on EVA-CLIP + FAISS."""

    def __init__(
        self,
        img_index_path: str,
        img_index_json_path: str,
        top_k: int = 1,
        device: str | None = None,
        ef_search: int | None = None,
    ):
        """Parameters"""
        self.top_k = top_k
        self.device = torch.device(device if device else "cpu")
        self.ef_search = ef_search

        self._img_index_path = img_index_path
        self._img_index_json_path = img_index_json_path

        self._lock = threading.Lock()
        self.img_index = None
        self.img_values = None
        self.processor = None
        self.embedding_model = None

    def _ensure_index(self):
        """Load the FAISS index and its JSON mapping (once)."""
        with self._lock:
            if self.img_index is not None:
                return

            print("Loading FAISS index …")
            self.img_index = faiss.read_index(self._img_index_path, faiss.IO_FLAG_MMAP)
            with open(self._img_index_json_path, "r") as f:
                self.img_values = json.load(f)

            if self.ef_search:
                try:
                    idx = faiss.downcast_index(self.img_index)
                    if hasattr(idx, "hnsw"):
                        idx.hnsw.efSearch = self.ef_search
                        print(f"HNSW efSearch set to {self.ef_search}.")
                except Exception as e:
                    print(f"Could not set efSearch: {e}")

            print(f"FAISS index loaded ({self.img_index.ntotal} vectors).")

    def _ensure_model(self):
        """Load the EVA-CLIP model and image processor (once)."""
        with self._lock:
            if self.embedding_model is not None:
                return

            self._model_dtype = torch.float16

            print("Loading EVA-CLIP embedding model …")
            self.processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.text_tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.embedding_model = (
                AutoModel.from_pretrained(
                    "BAAI/EVA-CLIP-8B",
                    dtype=self._model_dtype,
                    trust_remote_code=True,
                )
                .to(self.device)
                .eval()
            )

            for emb in (
                self.embedding_model.vision_model.embeddings,
                self.embedding_model.text_model.embeddings,
            ):
                n = emb.position_embedding.num_embeddings
                emb.position_ids = torch.arange(n, device=self.device).expand((1, -1))
            print(f"EVA-CLIP model loaded on {self.device} ({self._model_dtype}).")

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode an image into a normalised embedding vector."""
        self._ensure_model()

        image_tensor = self.processor(image, return_tensors="pt").pixel_values.to(
            self.device, dtype=self._model_dtype
        )

        with torch.no_grad():
            image_features = self.embedding_model.encode_image(image_tensor)

        image_features = F.normalize(image_features, dim=-1)
        return image_features.cpu().numpy().astype(np.float32)

    def encode_text(self, query: str) -> np.ndarray:
        """Encode a text query into the shared CLIP space (a ``(1, D)`` array)."""
        self._ensure_model()

        tokens = self.text_tokenizer(
            [query],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_features = self.embedding_model.encode_text(
                input_ids=tokens.input_ids, attention_mask=tokens.attention_mask
            )

        text_features = F.normalize(text_features, dim=-1)
        return text_features.cpu().numpy().astype(np.float32)

    def retrieve(self, image: Image.Image, question: str | None = None) -> list[dict]:
        """Articles whose reference images are nearest to ``image``."""
        return self.search_index(self.encode_image(image), self.top_k)

    def search_index(self, embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        """Search FAISS with a normalised ``(1, D)`` embedding."""
        self._ensure_index()
        distances, indices = self.img_index.search(embedding, k=top_k)

        results, seen = [], set()
        for idx, score in zip(indices[0], distances[0].tolist()):
            if idx == -1 or idx >= len(self.img_values):
                continue
            url, title, image_path = self.img_values[idx]
            if url in seen:
                continue
            seen.add(url)
            results.append({"wiki_url": url, "title": title,
                            "image_path": image_path, "score": score})
        return results
