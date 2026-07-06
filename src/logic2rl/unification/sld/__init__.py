"""The vectorized single-step SLD engine package.

  kb.py          the indexed logic program (FactIndex / RuleIndex / KB + branching budgets)
  resolution.py  unification + fact/rule resolution + variable standardization
  engine.py      :class:`SLD` — consult (ctor) / ``derive`` (successor fn) / ``prove`` (oracle)
"""
from logic2rl.unification.sld.engine import SLD
from logic2rl.unification.sld.kb import KB, FactIndex, RuleIndex

__all__ = ["SLD", "KB", "FactIndex", "RuleIndex"]
