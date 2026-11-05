"""Dense vector index and neural embedding model for semantic search."""

from typing import Any, List, Tuple, cast

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class VectorIndex:
    """Dense vector index supporting cosine similarity top-k search."""

    def __init__(self, embeddings: np.ndarray) -> None:
        """Initialize vector index with document embedding matrix.

        Args:
            embeddings: 2D numpy array of normalized document embeddings.
        """
        self.__embeddings: np.ndarray = embeddings

    @property
    def embeddings(self) -> np.ndarray:
        """Return the internal document embeddings array.

        Returns:
            2D numpy array of document embeddings.
        """
        return self.__embeddings

    def search(
        self, query_vector: np.ndarray, k: int = 5
    ) -> List[Tuple[int, float]]:
        """Search top-k nearest embeddings by dot product similarity.

        Args:
            query_vector: 1D normalized query embedding vector.
            k: Maximum number of top matches to return.

        Returns:
            List of (document_index, cosine_similarity) tuples.
        """
        if k <= 0 or self.__embeddings.size == 0:
            return []
        scores: np.ndarray = np.dot(self.__embeddings, query_vector)
        top_k: int = min(k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        sorted_top = top_indices[np.argsort(-scores[top_indices])]
        return [(int(i), float(scores[i])) for i in sorted_top]


class EmbeddingModel:
    """Neural embedding model wrapping Hugging Face transformers."""

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        """Load model and tokenizer for dense text representations.

        Args:
            model_name: Hugging Face pretrained model repository identifier.
        """
        self.__tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.__model: Any = AutoModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        self.__dimension: int = self.__model.config.hidden_size
        cast(torch.nn.Module, self.__model).eval()

    def __mean_pooling(
        self, output: Any, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention-masked mean pooling over token embeddings.

        Args:
            output: Model forward pass output containing token embeddings.
            mask: Attention mask tensor indicating non-padding tokens.

        Returns:
            Averaged sentence representation tensor.
        """
        toks: torch.Tensor = output[0]
        expanded: torch.Tensor = (
            mask.unsqueeze(-1).expand(toks.size()).float()
        )
        summed: torch.Tensor = torch.sum(toks * expanded, 1)
        clamped: torch.Tensor = torch.clamp(expanded.sum(1), min=1.0)
        return summed / clamped

    def __encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode and normalize a batch of text strings into unit vectors.

        Args:
            texts: List of text strings to embed in a batch.

        Returns:
            2D numpy array of L2-normalized embedding vectors.
        """
        enc = self.__tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        with torch.inference_mode():
            out = self.__model(**enc)
            pooled: torch.Tensor = self.__mean_pooling(
                out, enc["attention_mask"]
            )
            norm: torch.Tensor = torch.nn.functional.normalize(pooled)
            return norm.numpy().astype(np.float32)

    def encode(
        self, texts: List[str], batch_size: int = 16
    ) -> np.ndarray:
        """Encode multiple texts in mini-batches with a progress indicator.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts per mini-batch.

        Returns:
            2D numpy array of L2-normalized embeddings for all texts.
        """
        if not texts:
            return np.empty((0, self.__dimension), dtype=np.float32)
        batches: List[np.ndarray] = []
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Embedding",
            unit="batch",
            leave=False,
        ):
            batch: List[str] = texts[i:i + batch_size]
            batches.append(self.__encode_batch(batch))
        return np.vstack(batches)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a normalized 1D vector.

        Args:
            query: Search query text string.

        Returns:
            1D normalized embedding vector for the query.
        """
        res: np.ndarray = self.__encode_batch([query])
        return cast(np.ndarray, res[0])
