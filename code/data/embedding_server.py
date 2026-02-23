"""
Multi-process embedding server for TensorFlow and HuggingFace models.

This module provides a robust, process-isolated embedding server that generates
text embeddings using HuggingFace transformer models. The server runs in a
separate process to avoid TensorFlow eager execution conflicts and provides
a clean interface for batch embedding generation.

Key Features:
- Process isolation to prevent TensorFlow graph/eager execution conflicts
- Automatic cleanup and resource management
- Batch processing with configurable padding and truncation
- Mean pooling over non-padding tokens for sequence-level embeddings
- Robust error handling with detailed traceback propagation
- Signal handling for graceful shutdown

Classes:
    EmbeddingServer: Main server class providing embedding generation interface
    
Functions:
    _worker: Internal worker process function for model execution
"""

import sys
import signal
import traceback
import multiprocessing as mp
import atexit

import numpy as np

from typing import Any, Dict, List, Optional, Union

class EmbeddingServer:
    """
    Process-isolated embedding server for HuggingFace transformer models.
    
    Provides a clean interface for generating text embeddings using transformer models
    while avoiding TensorFlow graph/eager execution conflicts. The server runs the
    model in a separate process and communicates via multiprocessing queues.
    
    The server automatically handles:
    - Model loading and initialization in the worker process
    - Batch processing with padding and attention masking
    - Mean pooling over non-padding tokens for sequence representations
    - Error propagation with detailed tracebacks
    - Resource cleanup and process management
    
    Attributes:
        model_name (str): HuggingFace model identifier or local path
        req_q (mp.Queue): Request queue for sending embedding requests
        res_q (mp.Queue): Response queue for receiving embedding results
        p (mp.Process): Worker process running the embedding model
        _closed (bool): Flag indicating if the server has been shut down
        
    Example:
        >>> server = EmbeddingServer("bert-base-uncased")
        >>> questions = [["What", "is", "AI", "?"], ["How", "does", "it", "work", "?"]]
        >>> embeddings = server.embed(questions, pad_id=0, cls_id=101, sep_id=102)
        >>> print(embeddings.shape)  # (2, 768) for BERT-base
        >>> server.close()
        
    Note:
        Always call close() when done, or use as a context manager to ensure
        proper cleanup of the worker process and resources.
    """
    def __init__(self, model_name: str, start_method: str = "spawn", max_queue_size: int = 8) -> None:
        """
        Initialize the embedding server and start the worker process.
        
        Creates multiprocessing queues for communication and starts a worker process
        that loads the specified transformer model. Sets up signal handlers and
        cleanup routines for graceful shutdown.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "bert-base-uncased") 
                       or path to local model directory
            start_method: Multiprocessing start method ("spawn", "fork", "forkserver").
                         "spawn" is recommended for TensorFlow/HuggingFace compatibility
            max_queue_size: Maximum number of items that can be queued for processing
            
        Raises:
            OSError: If the worker process fails to start
            ValueError: If the model_name is invalid or model cannot be loaded
        """
        self.model_name: str = model_name
        self._closed: bool = False
        
        # Create multiprocessing context and queues
        ctx = mp.get_context(start_method)  # safer than fork for TF/HF
        self.req_q: mp.Queue = ctx.Queue(maxsize=max_queue_size)
        self.res_q: mp.Queue = ctx.Queue(maxsize=max_queue_size)
        
        # Start worker process
        self.p: mp.Process = ctx.Process(
            target=_worker, 
            args=(self.model_name, self.req_q, self.res_q), 
            daemon=True
        )
        self.p.start()

        # Ensure cleanup on interpreter shutdown
        atexit.register(self.close)

        # Set up signal handlers for graceful shutdown (best effort)
        try:
            signal.signal(signal.SIGINT, self._signal_close)
            signal.signal(signal.SIGTERM, self._signal_close)
        except Exception:
            # Signal handling may fail in some environments (e.g., threads)
            pass

    def _signal_close(self, *args) -> None:
        """Handle shutdown signals by closing the server."""
        self.close()

    def __enter__(self) -> 'EmbeddingServer':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()

    def embed(
        self, 
        token_id_batches: Union[List[List[int]], List[str]], 
        pad_id: int, 
        cls_id: int, 
        sep_id: int, 
        max_length: int = 128, 
        timeout: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of tokenized sequences.
        
        Sends a batch of token sequences to the worker process for embedding
        generation. The worker handles tokenization preprocessing (adding special
        tokens, padding), model inference, and mean pooling to produce sequence-level
        embeddings.
        
        Args:
            token_id_batches: List of token ID sequences or text strings to embed.
                            If strings are provided, they should be pre-tokenized text
            pad_id: Token ID used for padding sequences to uniform length
            cls_id: Token ID for the [CLS] classification token (typically 101 for BERT)
            sep_id: Token ID for the [SEP] separator token (typically 102 for BERT)  
            max_length: Maximum sequence length; longer sequences will be truncated
            timeout: Maximum time to wait for embedding generation (seconds).
                    None means wait indefinitely
                    
        Returns:
            Numpy array of shape (batch_size, embedding_dim) containing the
            mean-pooled embeddings for each input sequence
            
        Raises:
            RuntimeError: If the server has been closed or if the worker process
                         encounters an error during embedding generation
            TimeoutError: If the operation times out (when timeout is specified)
            
        Example:
            >>> server = EmbeddingServer("bert-base-uncased")
            >>> sequences = [[2054, 2003, 6790], [2129, 2515, 2009, 2147]]  # "what is ai", "how does it work"
            >>> embeddings = server.embed(sequences, pad_id=0, cls_id=101, sep_id=102)
            >>> print(embeddings.shape)  # (2, 768)
        """
        if self._closed:
            raise RuntimeError("EmbeddingServer is closed. Cannot generate embeddings.")
        
        # Prepare request payload
        payload: Dict[str, Any] = {
            "token_id_batches": token_id_batches,
            "pad_id": pad_id,
            "cls_id": cls_id,
            "sep_id": sep_id,
            "max_length": max_length,
        }
        
        # Send request and wait for response
        self.req_q.put(payload)
        result = self.res_q.get(timeout=timeout) if timeout else self.res_q.get()
        
        # Handle worker errors
        if isinstance(result, dict) and "_error" in result:
            error_msg = result['_error']
            traceback_info = result.get('_traceback', '')
            raise RuntimeError(f"Embedding worker error: {error_msg}\n{traceback_info}")
            
        return result  # np.ndarray of shape (batch_size, embedding_dim)

    def close(self) -> None:
        """
        Shut down the embedding server and clean up all resources.
        
        Performs a graceful shutdown by:
        1. Sending termination signal to worker process
        2. Waiting for process to finish (with timeout)
        3. Force-terminating if necessary
        4. Closing and cleaning up multiprocessing queues
        
        This method is idempotent and safe to call multiple times.
        It's automatically called on interpreter shutdown and by the
        context manager protocol.
        """
        if self._closed:
            return
        self._closed = True

        # Request worker shutdown
        try:
            self.req_q.put_nowait(None)  # None signals worker to exit
        except Exception:
            pass  # Queue might be full, worker will be terminated anyway

        # Wait for graceful shutdown
        if self.p.is_alive():
            self.p.join(timeout=3.0)

        # Force termination if still running
        if self.p.is_alive():
            try:
                self.p.terminate()
            except Exception:
                pass
            self.p.join(timeout=2.0)

        # Clean up multiprocessing queues
        for q in (self.req_q, self.res_q):
            try:
                q.close()
            except Exception:
                pass
            try:
                q.join_thread()
            except Exception:
                pass


def _worker(model_name: str, req_q: mp.Queue, res_q: mp.Queue) -> None:
    """
    Worker process function for embedding generation.
    
    Runs in a separate process to isolate TensorFlow model execution and avoid
    graph/eager execution conflicts. Loads the specified HuggingFace model and
    processes embedding requests from the main process.
    
    Process workflow:
    1. Load transformer model with TensorFlow eager execution
    2. Listen for requests on the request queue
    3. For each request:
       - Preprocess sequences (add special tokens, pad to uniform length)
       - Generate attention masks for non-padding tokens
       - Run model inference to get hidden states
       - Apply mean pooling over non-padding tokens
       - Return embeddings via response queue
    4. Handle errors gracefully with detailed tracebacks
    5. Clean up TensorFlow session on exit
    
    Args:
        model_name: HuggingFace model identifier or local path
        req_q: Multiprocessing queue for receiving embedding requests
        res_q: Multiprocessing queue for sending embedding results
        
    Request Format:
        Dict with keys: token_id_batches, pad_id, cls_id, sep_id, max_length
        
    Response Format:
        - Success: np.ndarray of shape (batch_size, embedding_dim)
        - Error: Dict with "_error" and "_traceback" keys
        
    Note:
        This function runs in a separate process and should not be called directly.
        Communication happens only through the multiprocessing queues.
    """
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    try:
        import tensorflow as tf
        from transformers import TFAutoModel

        # Load model with TensorFlow eager execution enabled
        model = TFAutoModel.from_pretrained(model_name, from_pt=False)

        while True:
            # Wait for next request
            item = req_q.get()
            if item is None:  # Shutdown signal
                break

            # Extract request parameters
            token_id_batches: List[List[int]] = item["token_id_batches"]
            pad_id: int = item["pad_id"]
            cls_id: int = item["cls_id"] 
            sep_id: int = item["sep_id"]
            max_length: int = item.get("max_length", 128)

            # Preprocess sequences: add special tokens and truncate
            seqs: List[List[int]] = []
            for ids in token_id_batches:
                x = list(ids)
                # Add [CLS] token at start if missing
                if len(x) == 0 or x[0] != cls_id: 
                    x = [cls_id] + x
                # Add [SEP] token at end if missing  
                if len(x) == 0 or x[-1] != sep_id: 
                    x = x + [sep_id]
                seqs.append(x[:max_length])  # Truncate to max_length

            # Handle empty batch
            if not seqs:
                empty_result = np.zeros((0, model.config.hidden_size), dtype=np.float32)
                res_q.put(empty_result)
                continue

            # Create padded input arrays
            max_len = min(max(len(s) for s in seqs), max_length)
            input_ids = np.full((len(seqs), max_len), pad_id, dtype=np.int32)
            attn_mask = np.zeros((len(seqs), max_len), dtype=np.int32)
            
            # Fill arrays with sequence data
            for i, s in enumerate(seqs):
                seq_len = min(len(s), max_len)
                input_ids[i, :seq_len] = s[:seq_len]
                attn_mask[i, :seq_len] = 1

            # Run model inference
            outputs = model(
                input_ids=tf.convert_to_tensor(input_ids),
                attention_mask=tf.convert_to_tensor(attn_mask)
            )
            
            # Extract hidden states and apply mean pooling
            last_hidden = outputs.last_hidden_state                 # Shape: [batch_size, seq_len, hidden_size]
            mask = tf.cast(attn_mask[:, :, None], tf.float32)       # Shape: [batch_size, seq_len, 1]
            
            # Mean pooling over non-padding tokens
            summed = tf.reduce_sum(last_hidden * mask, axis=1)      # Shape: [batch_size, hidden_size]
            lengths = tf.reduce_sum(mask, axis=1)                   # Shape: [batch_size, 1]
            final_embeddings = summed / tf.maximum(lengths, 1e-9)   # Avoid division by zero
            
            # Send results back to main process
            res_q.put(final_embeddings.numpy())
            
    except Exception as e:
        # Handle any errors during processing
        tb = traceback.format_exc()
        error_payload = {"_error": str(e), "_traceback": tb}
        try:
            res_q.put(error_payload)
        except Exception:
            pass  # Queue might be closed
    finally:
        # Clean up TensorFlow session
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass