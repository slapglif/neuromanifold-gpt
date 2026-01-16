"""
Neuro-Symbolic Wave Manifold Network (NS-WMN) / WaveManifoldGPT.

The unified architecture realizing the "2026 Vision":
- FNO Input
- Mamba + Soliton Backbone
- Topological Regularization
- Continuous Generation Output
"""

import torch
import torch.nn as nn
from neuromanifold_gpt.config.wave_manifold_config import WaveManifoldConfig
from neuromanifold_gpt.model.wave_manifold_block import WaveManifoldBlock
from neuromanifold_gpt.model.fno.multimodal_encoder import MultimodalFNOEncoder
from neuromanifold_gpt.model.continuous.output_head import ContinuousOutputHead
from neuromanifold_gpt.model.topology.topological_head import TopologicalHead

class WaveManifoldGPT(nn.Module):
    def __init__(self, config: WaveManifoldConfig):
        super().__init__()
        self.config = config
        
        # 1. Input Layer
        if config.use_fno_encoder:
            self.input_encoder = MultimodalFNOEncoder(
                vocab_size=config.vocab_size if hasattr(config, 'vocab_size') else 50304,
                embed_dim=config.n_embd,
                fno_modes=config.fno_modes
            )
        else:
            self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
            self.drop_emb = nn.Dropout(config.dropout)

        # 2. Backbone (Pure Wave/Hyena/Mamba - No Hybrid Attention)
        self.blocks = nn.ModuleList([
            WaveManifoldBlock(config, max_seq_len=config.block_size) for _ in range(config.n_layer)
        ])
        self.norm_f = nn.LayerNorm(config.n_embd)

        # 3. Output Heads
        # A. Discrete Head (Standard)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # B. Continuous Head (Rectified Flow)
        if config.use_continuous_head:
            # Override dimensions to match model
            config.continuous_head_config.embed_dim = config.n_embd
            # Optional: Scale hidden dim relative to embed dim if not set explicitly
            # For now, trust the config defaults or user overrides
            self.continuous_head = ContinuousOutputHead(config.continuous_head_config)
            
        # C. Topological Head (Regularization)
        if config.use_topological_loss:
            self.topo_head = TopologicalHead(config.n_embd)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 1. Input Encoding
        if self.config.use_fno_encoder:
            x = self.input_encoder(idx)
        else:
            tok_emb = self.token_emb(idx)
            pos_emb = self.pos_emb[:, :T, :]
            x = self.drop_emb(tok_emb + pos_emb)
            
        # 2. Backbone Processing
        for block in self.blocks:
            x = block(x)
            
        x = self.norm_f(x)
        
        # 3. Heads & Loss
        logits = self.lm_head(x)
        
        loss = None
        info = {}
        
        if targets is not None:
            # Discrete Loss (Standard CE)
            loss_discrete = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )
            loss = loss_discrete
            info['loss_discrete'] = loss_discrete.item()
            
            # Continuous Loss (Flow Matching)
            if self.config.use_continuous_head:
                # Target for continuous head is the embedding of the *next* token
                # This requires looking up embeddings of targets. 
                # For simplicity, we can use the model's own token embeddings as ground truth
                # or project targets to embedding space.
                
                # Assuming input_encoder or token_emb gives us "clean" embeddings.
                with torch.no_grad():
                    if self.config.use_fno_encoder:
                        target_embeds = self.input_encoder(targets)
                    else:
                        target_embeds = self.token_emb(targets)
                
                cont_out = self.continuous_head.compute_loss(target_embeds, condition=x)
                loss_cont = cont_out['continuous_loss']
                
                # Add to total loss
                loss = loss + loss_cont
                info['loss_continuous'] = loss_cont.item()
                
            # Topological Loss
            if self.config.use_topological_loss:
                topo_loss, topo_info = self.topo_head(x)
                loss = loss + self.config.topology_weight * topo_loss
                info.update(topo_info)
                
        return logits, loss, info
