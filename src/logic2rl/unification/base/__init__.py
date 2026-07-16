"""Shared substrate for the unification engines.

  kb.py          the indexed logic program (FactIndex / RuleIndex / KB + branching budgets)
  resolution.py  unification primitives + fact/rule resolution + var standardization
  soft.py        ``fill_vars`` — commit a chosen variable assignment (the fill primitive)
  joint.py       :class:`FactJoint` — all real-fact groundings of open-var states (the fact joint)
  engine.py      :class:`BaseEngine` — consult / ``derive`` / ``replace_candidates`` / ``prove``

The concrete engines (``sld.SLD``, ``enumerate.Enumerate``) extend :class:`BaseEngine`; they
differ in ``derive`` and the available ``*_fill_vars`` methods at the ``replace_candidates`` seam.
"""
from logic2rl.unification.base.engine import BaseEngine
from logic2rl.unification.base.joint import FactJoint
from logic2rl.unification.base.kb import (KB, FactIndex, RuleIndex, fact_contains,
                                          is_const, is_var, pack_atoms)
from logic2rl.unification.base.soft import fill_vars

__all__ = ["BaseEngine", "FactJoint", "KB", "FactIndex", "RuleIndex", "fact_contains",
           "is_const", "is_var", "pack_atoms", "fill_vars"]
