"""Shared substrate for the unification engines.

  kb.py          the indexed logic program (FactIndex / RuleIndex / KB + branching budgets)
  resolution.py  unification primitives + fact/rule resolution + var standardization
  soft.py        soft unification (``resolve_soft_facts``) — the neural open-var grounding
  engine.py      :class:`BaseEngine` — consult / ``derive`` / pack / prune / ``prove``

The concrete engines (``sld.SLD``, ``join.Join``) extend :class:`BaseEngine`; they differ only
in ``resolve_soft_facts``.
"""
from logic2rl.unification.base.engine import BaseEngine
from logic2rl.unification.base.kb import (KB, FactIndex, RuleIndex, fact_contains,
                                          is_const, is_var, pack_atoms)
from logic2rl.unification.base.soft import resolve_soft_facts

__all__ = ["BaseEngine", "KB", "FactIndex", "RuleIndex", "fact_contains",
           "is_const", "is_var", "pack_atoms", "resolve_soft_facts"]
