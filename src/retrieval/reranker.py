import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderReranker:
    """Cross-encoder paragraph reranker."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str | None = None,
        max_length: int = 512,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.last_top_score: float | None = None

        print(f"Loading cross-encoder reranker {model_name} ({dtype}) …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        print(f"Cross-encoder loaded on {self.device}.")

    @torch.inference_mode()
    def rerank(
        self, query: str, paragraphs: list[str], top_n: int = 3,
        batch_size: int = 16, force_sort: bool = False,
    ) -> list[str]:
        """Return the *top_n* paragraphs most relevant to the query."""
        if not paragraphs:
            return []
        if len(paragraphs) <= top_n and not force_sort:
            return paragraphs

        scores: list[float] = []
        for i in range(0, len(paragraphs), batch_size):
            batch = paragraphs[i : i + batch_size]
            inputs = self.tokenizer(
                [[query, p] for p in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs).logits.view(-1)
            scores.extend(logits.float().tolist())

        order = sorted(range(len(paragraphs)), key=lambda i: scores[i], reverse=True)
        self.last_top_score = scores[order[0]] if order else None
        return [paragraphs[i] for i in order[:top_n]]

    def score_of_best(self, query: str, paragraphs: list[str], **kw) -> float | None:
        """Relevance of the best paragraph in the pool, on the model's own scale."""
        self.rerank(query, paragraphs, top_n=1, force_sort=True, **kw)
        return self.last_top_score
