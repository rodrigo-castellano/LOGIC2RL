"""Shared substrate for the unification engines.

  kb.py          the indexed logic program (FactIndex / RuleIndex / KB + branching budgets)
  resolution.py  unification primitives + fact/rule resolution + var standardization
  engine.py      :class:`BaseEngine` — consult / ``derive`` / the ``replace_candidates``
                 seam (delegates to an app-attached candidate filler) / ``prove``

The concrete engines (``sld.SLD``, ``enumerate.Enumerate``) extend :class:`BaseEngine` and
differ only in ``derive``. All open-var FILL logic (soft/hard commits, discards) lives in the
app's filler object, attached as ``engine.candidate_filler``.
"""
from logic2rl.unification.base.engine import BaseEngine
from logic2rl.unification.base.kb import (KB, FactIndex, RuleIndex, fact_contains,
                                          is_const, is_var, pack_atoms)

__all__ = ["BaseEngine", "KB", "FactIndex", "RuleIndex", "fact_contains",
           "is_const", "is_var", "pack_atoms"]
