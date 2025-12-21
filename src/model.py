"""HybridTRM model with RoPE and latent reasoning loop."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from .config import ModelConfig


# ============================================================================
# RoPE (Rotary Position Embeddings) Implementation
# ============================================================================


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    Args:
        dim: Dimension of the embeddings (must be even)
        end: Maximum sequence length
        theta: Base for the geometric progression

    Returns:
        Tensor of shape (end, dim//2, 2) representing cos and sin
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # Return as (seq_len, dim//2, 2) where last dim is [cos, sin]
    freqs_cis = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch, seq_len, n_head, head_dim)
        freqs_cis: Precomputed frequencies of shape (seq_len, head_dim//2, 2)

    Returns:
        Tensor with rotary embeddings applied
    """
    # x: (batch, seq_len, n_head, head_dim)
    batch, seq_len, n_head, head_dim = x.shape

    # Reshape x into pairs for rotation
    x_reshaped = x.float().reshape(batch, seq_len, n_head, head_dim // 2, 2)

    # freqs_cis: (seq_len, head_dim//2, 2) -> (1, seq_len, 1, head_dim//2, 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Apply rotation: x_out = x * cos + rotate(x) * sin
    cos = freqs_cis[..., 0]  # (1, seq_len, 1, head_dim//2)
    sin = freqs_cis[..., 1]  # (1, seq_len, 1, head_dim//2)

    # x_reshaped[..., 0] is real part, x_reshaped[..., 1] is imaginary part
    x0 = x_reshaped[..., 0]  # (batch, seq_len, n_head, head_dim//2)
    x1 = x_reshaped[..., 1]  # (batch, seq_len, n_head, head_dim//2)

    # Rotation formula
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    # Stack back and reshape
    out = torch.stack([out0, out1], dim=-1)
    out = out.reshape(batch, seq_len, n_head, head_dim)

    return out.type_as(x)


# ============================================================================
# Modified GPT2 Attention with RoPE
# ============================================================================


class GPT2AttentionWithRoPE(nn.Module):
    """GPT2 attention with RoPE instead of learned positional embeddings."""

    def __init__(self, config: GPT2Config, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        max_positions = config.n_positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                self.head_dim,
                max_positions * 2,  # Allow for longer sequences during inference
                theta=getattr(config, "rope_theta", 10000.0)
            ),
            persistent=False,
        )

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Split hidden_size dim into num_heads and head_dim."""
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merge num_heads and head_dim back into hidden_size."""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,  # CRITICAL: Allow bidirectional attention for reasoning
        **kwargs,  # Catch any other kwargs from newer transformers versions
    ):
        # Compute Q, K, V
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.split_size, dim=2)

        # Split heads: (batch, seq_len, n_head, head_dim)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Apply RoPE to Q and K
        # Reshape to (batch, seq_len, n_head, head_dim) for RoPE
        batch_size, num_heads, seq_len, head_dim = query.shape
        query = query.transpose(1, 2)  # (batch, seq_len, n_head, head_dim)
        key = key.transpose(1, 2)

        # Get frequencies for current sequence length
        freqs_cis = self.freqs_cis[:seq_len]

        query = apply_rotary_emb(query, freqs_cis)
        key = apply_rotary_emb(key, freqs_cis)

        # Transpose back: (batch, n_head, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)

        # Compute attention
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # CRITICAL FIX: Conditional masking for bidirectional reasoning
        # During refinement (use_causal_mask=False), latent tokens can attend to all tokens
        # During autoregressive generation (use_causal_mask=True), enforce causality
        if use_causal_mask:
            causal_mask = self.bias[:, :, :seq_len, :seq_len]
            attn_weights = torch.where(
                causal_mask,
                attn_weights,
                torch.tensor(-1e4, dtype=attn_weights.dtype, device=attn_weights.device)
            )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply head mask if provided
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # GPT2Block expects (output, weights) or (output, weights, cache)
        # Always return weights (or None if not needed)
        outputs = (attn_output, attn_weights if output_attentions else None)

        if use_cache:
            outputs = outputs + (None,)  # We don't use KV cache with RoPE in this implementation

        return outputs


# ============================================================================
# GPT2 Model with RoPE (No Learned Positional Embeddings)
# ============================================================================


class GPT2ModelWithRoPE(nn.Module):
    """GPT2 model with RoPE instead of learned positional embeddings."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        # NOTE: No wpe (learned positional embeddings) - using RoPE instead

        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer blocks with RoPE attention
        self.h = nn.ModuleList([
            self._make_block_with_rope(config, i) for i in range(config.num_hidden_layers)
        ])

        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Gradient checkpointing
        self.gradient_checkpointing = False

    def _make_block_with_rope(self, config: GPT2Config, layer_idx: int):
        """Create a GPT2 block with RoPE attention wrapped for use_causal_mask support."""
        # We need to manually construct the block with our custom attention
        block = GPT2Block(config, layer_idx=layer_idx)
        # Replace the attention module
        block.attn = GPT2AttentionWithRoPE(config, layer_idx=layer_idx)

        # Store original forward for wrapping
        original_forward = block.forward

        # Wrap the block's forward to intercept and pass use_causal_mask to attention
        def wrapped_forward(hidden_states, attention_mask=None, use_causal_mask=True, **kwargs):
            # Temporarily set use_causal_mask on the attention module
            # We'll pass it via kwargs to attention
            layer_past = kwargs.get('layer_past', None)
            head_mask = kwargs.get('head_mask', None)
            encoder_hidden_states = kwargs.get('encoder_hidden_states', None)
            encoder_attention_mask = kwargs.get('encoder_attention_mask', None)
            use_cache = kwargs.get('use_cache', False)
            output_attentions = kwargs.get('output_attentions', False)

            # Call attention with use_causal_mask
            residual = hidden_states
            hidden_states = block.ln_1(hidden_states)
            attn_output, attn_weights = block.attn(
                hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                use_causal_mask=use_causal_mask,
            )
            hidden_states = attn_output + residual

            # Feed-forward
            residual = hidden_states
            hidden_states = block.ln_2(hidden_states)
            feed_forward_hidden_states = block.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (attn_weights,)
            if use_cache:
                outputs += (None,)  # No cache

            return outputs

        # Replace the forward method
        block.forward = wrapped_forward
        return block

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,  # CRITICAL: Control causal vs bidirectional attention
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # No positional embeddings added here - RoPE handles it in attention
        hidden_states = self.drop(inputs_embeds)

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Forward through transformer blocks
        for i, block in enumerate(self.h):
            if self.gradient_checkpointing and self.training:
                # For gradient checkpointing, we need to pass use_causal_mask via closure
                # Create a closure that captures use_causal_mask
                def create_custom_forward(block, use_causal_mask):
                    def custom_forward(hidden_states, attention_mask):
                        return block(hidden_states, attention_mask=attention_mask, use_causal_mask=use_causal_mask)[0]
                    return custom_forward

                hidden_states = self._gradient_checkpointing_func(
                    create_custom_forward(block, use_causal_mask),
                    hidden_states,
                    attention_mask,
                )
            else:
                outputs = block(hidden_states, attention_mask=attention_mask, use_causal_mask=use_causal_mask)
                hidden_states = outputs[0]

        hidden_states = self.ln_f(hidden_states)

        return hidden_states

    def _gradient_checkpointing_func(self, func, *args):
        """Wrapper for gradient checkpointing."""
        return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False)


# ============================================================================
# HybridTRM: Main Model
# ============================================================================


class HybridTRM(nn.Module):
    """
    Hybrid Transformer with Recursive Memory (TRM).

    Combines a GPT2-based transformer with latent reasoning tokens
    that are refined through multiple loops before autoregressive generation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Create GPT2 config
        gpt2_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.seq_len_x + config.seq_len_y + config.n_latents,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_inner=config.n_inner,
            activation_function="gelu_new",
            resid_pdrop=config.resid_pdrop,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=False,
        )

        # Add RoPE config
        gpt2_config.rope_theta = config.rope_theta

        # Base transformer (with RoPE if enabled)
        if config.use_rope:
            self.transformer = GPT2ModelWithRoPE(gpt2_config)
        else:
            from transformers import GPT2Model
            self.transformer = GPT2Model(gpt2_config)

        # Enable gradient checkpointing
        if config.gradient_checkpointing:
            self.transformer.gradient_checkpointing = True

        # Latent embeddings (learnable memory tokens)
        self.latent_embeddings = nn.Parameter(
            torch.randn(1, config.n_latents, config.n_embd) * 0.02
        )

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights with token embeddings (standard practice)
        self.lm_head.weight = self.transformer.wte.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward_refine(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Refinement pass: process inputs through transformer with BIDIRECTIONAL attention.

        CRITICAL: During refinement, latent tokens need to attend to ALL tokens (including future ones)
        to perform reasoning. This is the key difference from standard autoregressive models.

        Args:
            inputs_embeds: Combined [x, z] embeddings

        Returns:
            Updated hidden states
        """
        # CRITICAL FIX: Use bidirectional attention for reasoning
        outputs = self.transformer(inputs_embeds=inputs_embeds, use_causal_mask=False)
        return outputs

    def forward_ar(
        self,
        x_z_embeds: torch.Tensor,
        y_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Autoregressive pass: generate predictions for target sequence with CAUSAL attention.

        During AR generation, we need causal masking for the target tokens.
        Note: Ideally, [x, z] should be bidirectional and only [y] should be causal (prefix-LM),
        but for simplicity we use full causal masking here. The key reasoning already happened
        in forward_refine with bidirectional attention.

        Args:
            x_z_embeds: Combined [x, z] embeddings after refinement
            y_ids: Target token IDs

        Returns:
            Logits for target sequence
        """
        device = y_ids.device
        batch_size, y_len = y_ids.shape

        # Embed targets (no positional embeddings - RoPE handles it)
        y_embeds = self.transformer.wte(y_ids)

        # Concatenate: [x, z, y]
        combined_embeds = torch.cat([x_z_embeds, y_embeds], dim=1)

        # Forward pass with causal masking for autoregressive generation
        outputs = self.transformer(inputs_embeds=combined_embeds, use_causal_mask=True)

        # Get logits for target sequence only
        logits = self.lm_head(outputs[:, -y_len:, :])

        return logits

    def forward(
        self,
        x_ids: torch.Tensor,
        y_ids: torch.Tensor,
        n_sup: Optional[int] = None,
        t_loops: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full training forward pass with deep supervision.

        Args:
            x_ids: Input token IDs (batch, seq_len_x)
            y_ids: Target token IDs (batch, seq_len_y)
            n_sup: Number of supervision steps (default: self.config.n_sup)
            t_loops: Number of refinement loops per step (default: self.config.t_loops)

        Returns:
            total_loss: Aggregated loss across supervision steps
            metrics: Dictionary of metrics for logging
        """
        n_sup = n_sup or self.config.n_sup
        t_loops = t_loops or self.config.t_loops

        device = x_ids.device
        batch_size = x_ids.size(0)

        # Embed inputs (no positional embeddings - RoPE handles it)
        x_embeds = self.transformer.wte(x_ids)

        # Initialize latent embeddings
        z_embeds = self.latent_embeddings.expand(batch_size, -1, -1)

        total_loss = 0.0
        supervision_losses = []

        # Deep supervision loop
        for sup_step in range(n_sup):
            # Refinement loops
            current_input = torch.cat([x_embeds, z_embeds], dim=1)

            for t in range(t_loops):
                is_last_loop = (t == t_loops - 1)

                # Only compute gradients on last loop (memory efficient)
                with torch.set_grad_enabled(is_last_loop or not self.training):
                    output_embeds = self.forward_refine(current_input)
                    z_embeds = output_embeds[:, -self.config.n_latents:, :]
                    current_input = torch.cat([x_embeds, z_embeds], dim=1)

            # Autoregressive generation loss
            logits_y = self.forward_ar(current_input, y_ids)

            # Compute cross-entropy loss
            shift_logits = logits_y[:, :-1, :].contiguous()
            shift_labels = y_ids[:, 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )

            total_loss = total_loss + loss
            supervision_losses.append(loss.item())

            # Detach latents for next supervision step
            z_embeds = z_embeds.detach()

        # Average loss across supervision steps
        avg_loss = total_loss / n_sup

        metrics = {
            "loss": avg_loss.item(),
            "supervision_losses": supervision_losses,
            "avg_supervision_loss": sum(supervision_losses) / len(supervision_losses),
        }

        return avg_loss, metrics

    @torch.no_grad()
    def generate(
        self,
        x_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            x_ids: Input token IDs (batch, seq_len_x)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            Generated token IDs (batch, max_new_tokens)
        """
        device = x_ids.device
        batch_size = x_ids.size(0)

        # Embed inputs
        x_embeds = self.transformer.wte(x_ids)

        # Initialize and refine latents with BIDIRECTIONAL attention
        z_embeds = self.latent_embeddings.expand(batch_size, -1, -1)
        current_input = torch.cat([x_embeds, z_embeds], dim=1)

        for t in range(self.config.t_loops):
            output_embeds = self.forward_refine(current_input)
            z_embeds = output_embeds[:, -self.config.n_latents:, :]
            current_input = torch.cat([x_embeds, z_embeds], dim=1)

        # Autoregressive generation with CAUSAL attention
        generated = []
        context = current_input

        for _ in range(max_new_tokens):
            # Forward pass with causal masking for generation
            outputs = self.transformer(inputs_embeds=context, use_causal_mask=True)
            logits = self.lm_head(outputs[:, -1, :])  # (batch, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            generated.append(next_token)

            # Update context
            next_embed = self.transformer.wte(next_token)
            context = torch.cat([context, next_embed], dim=1)

        return torch.cat(generated, dim=1)
