"""SLD engine — SLD derivation with **soft** open-var resolution at the resolve_soft_facts hook.

:class:`SLD` is a sibling of :class:`~logic2rl.unification.join.Join` (both extend
:class:`~logic2rl.unification.base.engine.BaseEngine`, neither inherits the other). It IS the base
engine: the shared machinery (consult / ``derive`` / pack / prune / ``prove``) plus, when ``soft``
is on, ``BaseEngine.resolve_soft_facts`` grounds each free variable with its most likely neural filler
(the shared ``base.soft.resolve_soft_facts``) at the post-derive hook. Pure SLD (soft off) has no
KGE grounding and is rejected by the KGE app (open-var states cannot be scored)."""
from __future__ import annotations

from logic2rl.unification.base.engine import BaseEngine


class SLD(BaseEngine):
    """SLD derivation with soft open-var resolution at the hook — see the module docstring.

    All behaviour is ``BaseEngine``'s (soft at ``resolve_soft_facts``); this is the named,
    exported engine for the ``sld/`` package and the builder's default ``engine_cls``."""


__all__ = ["SLD"]
