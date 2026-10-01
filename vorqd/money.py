"""USD decimal strings ↔ atomic token integers, the one conversion.

Money on the coordinator API is a decimal USD string; atomic integers exist only
where they are signed or verified. ``tests/vectors/money-v1.json`` is the
contract: the grammar below, at most ``decimals`` fraction digits (refused, never
rounded), ``atomic = usd × 10^decimals``, and a canonical format with no trailing
zeros. A rate is USD per 1M units of work, so ``"0.05"`` at 6 decimals is the
on-chain rate 50000.
"""

from __future__ import annotations

import re

_USD_RE = re.compile(r"(0|[1-9][0-9]*)(?:\.([0-9]+))?")


def parse_usd(text: str, decimals: int) -> int:
    """``text`` as atomic units at ``decimals``; ``ValueError`` on anything else."""
    if not isinstance(text, str):
        raise ValueError(f"{text!r} is not a USD decimal string")
    m = _USD_RE.fullmatch(text) if text.isascii() else None
    if m is None:
        raise ValueError(f"{text!r} is not a USD decimal string")
    whole, frac = m.group(1), m.group(2) or ""
    if len(frac) > decimals:
        raise ValueError(f"{text!r} has more than {decimals} fraction digits")
    return int(whole) * 10**decimals + int(frac.ljust(decimals, "0") or "0")


def format_usd(atomic: int, decimals: int) -> str:
    """Atomic units at ``decimals`` as the canonical USD string."""
    if isinstance(atomic, bool) or not isinstance(atomic, int) or atomic < 0:
        raise ValueError(f"{atomic!r} is not a non-negative atomic amount")
    whole, frac = divmod(atomic, 10**decimals)
    digits = str(frac).rjust(decimals, "0").rstrip("0")
    return f"{whole}.{digits}" if digits else str(whole)


__all__ = ("parse_usd", "format_usd")
