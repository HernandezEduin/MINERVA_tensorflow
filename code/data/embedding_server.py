import os
import multiprocessing as mp

import numpy as np
import tensorflow as tf
from transformers import TFAutoModel

from typing import List, Dict, Any

class EmbeddingServer:
    """
    EmbeddingServer
    ---------------
    Provides a simple interface for generating embeddings from tokenized text using a HuggingFace TF model.
    Runs the model in a separate process to avoid TensorFlow eager execution conflicts.
    """
    def __init__(self, model_name: str):
        """
        Initialize the embedding server and start the worker process.
        Args:
            model_name (str): HuggingFace model name or path.
        """
        self.model_name = model_name
        self.req_q: mp.Queue = mp.Queue(maxsize=4)
        self.res_q: mp.Queue = mp.Queue(maxsize=4)
        self.p = mp.Process(target=_worker, args=(self.model_name, self.req_q, self.res_q), daemon=True)
        self.p.start()
        self.model_name = model_name

    def embed(self, token_id_batches: List[List[int]], pad_id: int, cls_id: int, sep_id: int, max_length: int = 128) -> np.ndarray:
        """
        Request embeddings for a batch of tokenized sequences.
        Args:
            token_id_batches (List[List[int]]): List of token ID sequences.
            pad_id (int): Padding token ID.
            cls_id (int): CLS token ID.
            sep_id (int): SEP token ID.
            max_length (int): Maximum sequence length.
        Returns:
            np.ndarray: Array of pooled embeddings for each sequence.
        """
        payload: Dict[str, Any] = dict(
            token_id_batches=token_id_batches,
            pad_id=pad_id,
            cls_id=cls_id,
            sep_id=sep_id,
            max_length=max_length,
        )
        self.req_q.put(payload)
        arr = self.res_q.get()
        return arr

    def close(self):
        """
        Shut down the worker process cleanly.
        """
        try:
            self.req_q.put(None)
        except Exception:
            pass
        if self.p.is_alive():
            self.p.join(timeout=2)


def _worker(model_name: str, req_q: mp.Queue, res_q: mp.Queue):
    """
    Worker process that loads a HuggingFace TF model and computes embeddings for batches of token IDs.
    Listens for requests on req_q and returns results on res_q.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Eager ON here by default
    model = TFAutoModel.from_pretrained(model_name, from_pt=False)

    while True:
        item = req_q.get()
        if item is None:
            break
        # item: dict with token_id_batches, pad_id, cls_id, sep_id, max_length
        token_id_batches: List[List[int]] = item["token_id_batches"]
        pad_id = item["pad_id"]
        cls_id = item["cls_id"]
        sep_id = item["sep_id"]
        max_length = item.get("max_length", 128)

        # Build input_ids with special tokens ([CLS], [SEP]) and pad to uniform length
        seqs = []
        for ids in token_id_batches:
            x = list(ids)
            # Add [CLS] token at start if missing
            if len(x) == 0 or x[0] != cls_id:
                x = [cls_id] + x
            # Add [SEP] token at end if missing
            if len(x) == 0 or x[-1] != sep_id:
                x = x + [sep_id]
            x = x[:max_length]
            seqs.append(x)

        # Pad all sequences to max_len
        max_len = max(len(s) for s in seqs) if seqs else 1
        max_len = min(max_len, max_length)
        input_ids = np.full((len(seqs), max_len), pad_id, dtype=np.int32)
        attn_mask = np.zeros((len(seqs), max_len), dtype=np.int32)
        for i, s in enumerate(seqs):
            L = min(len(s), max_len)
            input_ids[i, :L] = s[:L]
            attn_mask[i, :L] = 1

        # Run model and mean-pool over non-padding tokens
        outputs = model(input_ids=tf.convert_to_tensor(input_ids),
                        attention_mask=tf.convert_to_tensor(attn_mask))
        last_hidden = outputs.last_hidden_state             # [B, L, H]
        mask = tf.cast(attn_mask[:, :, None], tf.float32)   # [B, L, 1]
        summed = tf.reduce_sum(last_hidden * mask, axis=1)  # [B, H]
        lengths = tf.reduce_sum(mask, axis=1)               # [B, 1]
        final = summed / tf.maximum(lengths, 1e-9)          # [B, H]
        res_q.put(final.numpy())                            # return numpy array