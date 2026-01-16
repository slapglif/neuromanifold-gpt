"""
lm-evaluation-harness adapter for NeuroManifoldGPT.

Provides a wrapper class that implements the lm_eval.api.model.LM interface,
enabling NeuroManifoldGPT models to be evaluated using the comprehensive
lm-evaluation-harness benchmark suite.

Usage:
    from neuromanifold_gpt.benchmarks.lm_eval_adapter import NeuroManifoldLM
    from lm_eval import evaluator

    # Create adapter from checkpoint
    lm = NeuroManifoldLM.from_checkpoint('out/ckpt.pt', device='cuda')

    # Run evaluation
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=['lambada_openai', 'hellaswag', 'piqa', 'winogrande'],
        num_fewshot=0,
    )

Requirements:
    pip install lm-eval

References:
    - lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
    - Model guide: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md
"""

import os
from typing import List, Dict, Tuple, Union, Optional, Any
import torch
import torch.nn.functional as F
import tiktoken

try:
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance
except ImportError:
    # Graceful degradation if lm-eval is not installed
    LM = object
    Instance = Any

from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.config.base import NeuroManifoldConfig


class NeuroManifoldLM(LM):
    """
    lm-evaluation-harness adapter for NeuroManifoldGPT.

    Wraps NeuroManifoldGPT models to provide the LM interface required by
    lm-evaluation-harness, enabling evaluation on comprehensive benchmark suites.

    Args:
        model: NeuroManifoldGPT model instance
        tokenizer: Tokenizer with encode/decode methods (tiktoken or custom)
        device: Device to run model on ('cuda' or 'cpu')
        batch_size: Batch size for evaluation (default: 1)
        max_length: Maximum sequence length (default: model.config.block_size)

    Example:
        >>> model = NeuroManifoldGPT(config)
        >>> tokenizer = tiktoken.get_encoding("gpt2")
        >>> lm = NeuroManifoldLM(model, tokenizer, device='cuda')
    """

    def __init__(
        self,
        model: NeuroManifoldGPT,
        tokenizer: Any,
        device: str = 'cuda',
        batch_size: int = 1,
        max_length: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length or model.config.block_size

        # Move model to device and set to eval mode
        self.model.to(self._device)
        self.model.eval()

        # Determine vocab size
        if hasattr(tokenizer, 'n_vocab'):
            self.vocab_size = tokenizer.n_vocab
        elif hasattr(tokenizer, 'vocab_size'):
            self.vocab_size = tokenizer.vocab_size
        else:
            self.vocab_size = model.config.vocab_size

        # Determine EOT token
        if hasattr(tokenizer, 'eot_token'):
            self.eot_token_id = tokenizer.eot_token
        elif hasattr(tokenizer, 'encode_ordinary'):
            # tiktoken tokenizer
            self.eot_token_id = tokenizer.encode_ordinary("<|endoftext|>")[0]
        else:
            self.eot_token_id = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cuda',
        batch_size: int = 1,
        max_length: Optional[int] = None,
    ) -> 'NeuroManifoldLM':
        """
        Load model from checkpoint and create adapter.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to run model on
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length

        Returns:
            NeuroManifoldLM adapter instance

        Example:
            >>> lm = NeuroManifoldLM.from_checkpoint('out/ckpt.pt', device='cuda')
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract config and create model
        config = checkpoint['config']
        model = NeuroManifoldGPT(config)

        # Load state dict
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        # Setup tokenizer
        # Try to load meta.pkl from dataset if available
        tokenizer = None
        if hasattr(config, 'dataset') and config.dataset:
            meta_path = os.path.join('data', config.dataset, 'meta.pkl')
            if os.path.exists(meta_path):
                import pickle
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                # Create custom tokenizer from meta
                class CustomTokenizer:
                    def __init__(self, stoi, itos):
                        self.stoi = stoi
                        self.itos = itos
                        self.vocab_size = len(itos)
                        self.eot_token = 0

                    def encode(self, text):
                        return [self.stoi.get(c, 0) for c in text]

                    def decode(self, tokens):
                        return ''.join([self.itos.get(t, '') for t in tokens])

                tokenizer = CustomTokenizer(meta['stoi'], meta['itos'])

        # Fallback to GPT-2 tokenizer
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("gpt2")

        return cls(model, tokenizer, device, batch_size, max_length)

    @property
    def device(self) -> str:
        """Device the model is running on."""
        return self._device

    @property
    def batch_size(self) -> int:
        """Batch size for evaluation."""
        return self._batch_size

    @property
    def max_length(self) -> int:
        """Maximum sequence length."""
        return self._max_length

    @property
    def tokenizer_name(self) -> str:
        """Tokenizer identifier for caching."""
        return "neuromanifold-gpt2"

    def tok_encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text)
        elif hasattr(self.tokenizer, 'encode_ordinary'):
            return self.tokenizer.encode_ordinary(text)
        else:
            raise NotImplementedError("Tokenizer must have encode or encode_ordinary method")

    def tok_decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(tokens)
        else:
            raise NotImplementedError("Tokenizer must have decode method")

    def _model_call(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Internal method to call model forward pass.

        Args:
            tokens: Input token tensor (batch_size, seq_len)

        Returns:
            Logits tensor (batch_size, seq_len, vocab_size)
        """
        with torch.no_grad():
            # NeuroManifoldGPT returns (logits, loss, auxiliary_losses)
            logits, _, _ = self.model(tokens)
        return logits

    def _collate(self, inputs: List[Tuple[str, str]]) -> Tuple[torch.Tensor, List[int]]:
        """
        Collate inputs into batched tensors.

        Args:
            inputs: List of (context, continuation) pairs

        Returns:
            Tuple of (batched_tokens, continuation_lengths)
        """
        # Tokenize all inputs
        tokenized = []
        cont_lens = []
        for context, continuation in inputs:
            context_tokens = self.tok_encode(context)
            continuation_tokens = self.tok_encode(continuation)

            # Truncate if too long
            total_len = len(context_tokens) + len(continuation_tokens)
            if total_len > self._max_length:
                # Truncate context to fit
                max_context_len = self._max_length - len(continuation_tokens)
                context_tokens = context_tokens[-max_context_len:]

            combined = context_tokens + continuation_tokens
            tokenized.append(combined)
            cont_lens.append(len(continuation_tokens))

        # Pad to max length in batch
        max_len = max(len(t) for t in tokenized)
        padded = []
        for tokens in tokenized:
            pad_len = max_len - len(tokens)
            # Pad on the left with EOT token
            padded_tokens = [self.eot_token_id] * pad_len + tokens
            padded.append(padded_tokens)

        batch_tensor = torch.tensor(padded, dtype=torch.long, device=self._device)
        return batch_tensor, cont_lens

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of target strings conditioned on context.

        For each request, computes the log probability of the continuation
        given the context, and whether it's the most probable continuation.

        Args:
            requests: List of Instance objects with context and continuation

        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        results = []

        # Process in batches
        for i in range(0, len(requests), self._batch_size):
            batch = requests[i:i + self._batch_size]

            # Extract context and continuation pairs
            inputs = [(req.args[0], req.args[1]) for req in batch]

            # Collate into batch
            batch_tokens, cont_lens = self._collate(inputs)

            # Get logits from model
            logits = self._model_call(batch_tokens)

            # Compute log-likelihoods for each item in batch
            for j, cont_len in enumerate(cont_lens):
                # Get the logits for continuation positions
                # continuation starts at position: seq_len - cont_len
                seq_len = batch_tokens.size(1)
                cont_start = seq_len - cont_len

                # Get continuation tokens
                cont_tokens = batch_tokens[j, cont_start:]

                # Get logits for positions just before each continuation token
                # Logits at position t predict token at position t+1
                cont_logits = logits[j, cont_start-1:seq_len-1, :]

                # Compute log probabilities
                log_probs = F.log_softmax(cont_logits, dim=-1)

                # Sum log probabilities for continuation tokens
                log_likelihood = 0.0
                is_greedy = True
                for k, token_id in enumerate(cont_tokens):
                    log_prob = log_probs[k, token_id].item()
                    log_likelihood += log_prob

                    # Check if this token is the greedy choice
                    greedy_token = torch.argmax(log_probs[k]).item()
                    if greedy_token != token_id:
                        is_greedy = False

                results.append((log_likelihood, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Compute rolling log-likelihood for perplexity evaluation.

        Calculates the log probability of the complete input text,
        conditioned only on the end-of-text token.

        Args:
            requests: List of Instance objects with input text

        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        results = []

        for req in requests:
            # Get the full text
            text = req.args[0]
            tokens = self.tok_encode(text)

            # Truncate if too long
            if len(tokens) > self._max_length:
                tokens = tokens[:self._max_length]

            # Convert to tensor
            input_tokens = torch.tensor([tokens[:-1]], dtype=torch.long, device=self._device)
            target_tokens = tokens[1:]

            # Get logits
            logits = self._model_call(input_tokens)

            # Compute log probabilities
            log_probs = F.log_softmax(logits[0], dim=-1)

            # Sum log probabilities
            log_likelihood = 0.0
            is_greedy = True
            for i, target_id in enumerate(target_tokens):
                log_prob = log_probs[i, target_id].item()
                log_likelihood += log_prob

                # Check if greedy
                greedy_token = torch.argmax(log_probs[i]).item()
                if greedy_token != target_id:
                    is_greedy = False

            results.append((log_likelihood, is_greedy))

        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate text completions for given prompts.

        Produces sampled text from the model using provided generation
        parameters (max length, stopping sequences, etc.).

        Args:
            requests: List of Instance objects with prompts and generation args

        Returns:
            List of generated text strings
        """
        results = []

        for req in requests:
            # Extract prompt and generation parameters
            context = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}

            # Parse generation parameters
            max_gen_toks = gen_kwargs.get('max_gen_toks', 256)
            until = gen_kwargs.get('until', [])
            temperature = gen_kwargs.get('temperature', 0.8)
            top_k = gen_kwargs.get('top_k', 200)

            # Tokenize prompt
            tokens = self.tok_encode(context)
            if len(tokens) > self._max_length - max_gen_toks:
                tokens = tokens[-(self._max_length - max_gen_toks):]

            # Convert to tensor
            x = torch.tensor([tokens], dtype=torch.long, device=self._device)

            # Generate tokens
            generated = []
            for _ in range(max_gen_toks):
                # Crop to block size if needed
                x_cond = x if x.size(1) <= self._max_length else x[:, -self._max_length:]

                # Get logits
                logits = self._model_call(x_cond)
                logits = logits[:, -1, :] / temperature

                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                x = torch.cat([x, next_token], dim=1)
                generated.append(next_token.item())

                # Check stopping conditions
                decoded = self.tok_decode(generated)
                if any(stop_seq in decoded for stop_seq in until):
                    break

            # Decode generated tokens
            generated_text = self.tok_decode(generated)

            # Truncate at stopping sequences
            for stop_seq in until:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.index(stop_seq)]

            results.append(generated_text)

        return results
