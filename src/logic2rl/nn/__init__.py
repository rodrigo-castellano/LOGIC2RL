"""base.nn — generic neural building blocks (pillar: base).

Low-level, domain-agnostic modules shared by the policy and the KGE embedder:
manual CUDA-graph-safe attention/GRU blocks (``_blocks``), the generic atom- and
state-level set encoders (``atom_embedders`` / ``state_embedders``), and the generic
embedder that composes them (``embeddings.EmbedderLearnable``). Imports nothing from
``algorithm`` or ``kge`` — the KGE application injects its scoring atom encoders via
``kge.nn`` (``kge.nn.embeddings.EmbedderLearnable`` subclasses the composer here).

(Masked categorical action distributions are a policy concern and live with the
algorithm at ``algorithm/policy/`` — both policies use SB3 ``CategoricalDistribution``.)
"""
from .atom_embedders import Emb_Atom_Factory
from .embeddings import ConstantEmbeddings, EmbedderLearnable, PredicateEmbeddings
from .state_embedders import Emb_State_Factory

__all__ = [
    "Emb_Atom_Factory",
    "Emb_State_Factory",
    "EmbedderLearnable",
    "ConstantEmbeddings",
    "PredicateEmbeddings",
]
