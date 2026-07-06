"""Feature Extractor — Converts index-based observations to dense embeddings.

Transforms [B, G, A, 3] index observations into [B, G, E] dense embeddings via
the shared atom embedder. Returns TensorDict features with named keys.
"""
import torch
import torch.nn as nn
from tensordict import TensorDict


class CustomCombinedExtractor(nn.Module):
    """Convert index observations [B, G, A, 3] into dense embeddings [B, G, E].

    Three forward paths: full (actor+critic), actor-only, critic-only.
    """
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder
        self.embed_dim = embedder.embedding_dim

    def _embed_obs(self, obs) -> torch.Tensor:
        """Embed the current-state atoms -> obs_emb [B, 1, E]."""
        return self.embedder.get_embeddings_batch(obs["sub_index"].to(torch.int32))

    def forward(self, obs) -> TensorDict:
        """Full forward: keys obs_emb, act_emb, action_mask."""
        obs_emb = self._embed_obs(obs)
        act_emb = self.embedder.get_embeddings_batch(obs["derived_sub_indices"].to(torch.int32))
        return TensorDict(
            {"obs_emb": obs_emb, "act_emb": act_emb, "action_mask": obs["action_mask"]},
            batch_size=obs_emb.shape[:1],
        )

    def forward_actor(self, obs) -> TensorDict:
        """Actor-only: keys obs_emb, act_emb, action_mask."""
        return self.forward(obs)

    def forward_critic(self, obs) -> TensorDict:
        """Critic-only: skips action embedding (most expensive). Key: obs_emb."""
        obs_emb = self._embed_obs(obs)
        return TensorDict({"obs_emb": obs_emb}, batch_size=obs_emb.shape[:1])
