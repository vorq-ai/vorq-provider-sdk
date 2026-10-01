"""The mapping template engine (mapping-reference §3).

Two operations, run once per job before each backend request:

* :func:`render` — substitute ``{dot.path}`` / ``{uuid}`` tokens (and expand any
  residual ``env:`` reference) inside a mapping fragment, which may be a string,
  an object, or a **top-level array**. A whole-string token preserves the
  resolved value's type; an embedded token stringifies.
* :func:`extract` — evaluate a ``$.``-prefixed JSONPath against a response body,
  returning a scalar or, for a wildcard path, a list.

``$.``-prefixed strings are extraction markers, not template tokens, so
:func:`render` leaves them untouched.
"""

from __future__ import annotations

import json
import re
import uuid

from jsonpath_ng.jsonpath import Child
from jsonpath_ng.ext import parse

from .config import resolve_env

_TOKEN = re.compile(r"\{([^{}]+)\}")


#: What a string holding an optional token that resolved to nothing leaves behind,
#: so the key or list element it was the value of can be removed rather than sent
#: empty. Never a value an operator could write, and never one that leaves
#: :func:`render`.
_DROP = object()


def render(node, ctx: dict):
    """Recursively render a mapping fragment against ``ctx``.

    ``ctx`` exposes the namespaces ``input.*``, ``job.*``, ``submit.*`` (only in
    a poll request), ``handle`` (the submit response's poll handle, in a poll
    request) and the flat backend params (``base_url``, ``api_key``, ``model``).
    """
    out = _render(node, ctx)
    if out is _DROP:
        # A key can be dropped and so can an element; a whole fragment cannot.
        raise KeyError(f"unresolved template token: {node}")
    return out


def _render(node, ctx: dict):
    if isinstance(node, str):
        return _render_str(node, ctx)
    # A value that was an optional token resolving to nothing is dropped, not sent
    # as null or as an empty string: an absent end frame must mean a request with
    # no such field, which is what the backend's own optional parameter expects.
    if isinstance(node, dict):
        rendered = ((k, _render(v, ctx)) for k, v in node.items())
        return {k: v for k, v in rendered if v is not _DROP}
    if isinstance(node, list):
        out: list = []
        for item in node:
            value = _render(item, ctx)
            if value is _DROP:
                continue
            # A wildcard token names a list; as an element it is spliced in, so one
            # list can hold a singular reference and a listed one side by side.
            whole = _TOKEN.fullmatch(item) if isinstance(item, str) else None
            if whole and _is_wildcard(whole.group(1)) and isinstance(value, list):
                out.extend(value)
            else:
                out.append(value)
        # Every element dropped: the list goes with them. `[]` is not the same
        # request as no field, and a backend that validates it refuses the first.
        return _DROP if node and not out else out
    return node


def _render_str(s: str, ctx: dict):
    s = resolve_env(s)  # expand any residual env:NAME (usually done at load)
    whole = _TOKEN.fullmatch(s)
    if whole:  # single token -> preserve the resolved value's type
        return _resolve(whole.group(1), ctx)
    # Embedded, one missing optional token drops the whole string. The form this
    # exists for is `data:{type?};base64,{b64?}`, and rendering the missing parts
    # as nothing would send `data:;base64,` — present and wrong, where absent is
    # what an optional field means.
    values = [_resolve(m.group(1), ctx) for m in _TOKEN.finditer(s)]
    if any(v is _DROP for v in values):
        return _DROP
    resolved = iter(values)
    return _TOKEN.sub(lambda _: str(next(resolved)), s)


def _resolve(token: str, ctx: dict):
    """One token's value, or :data:`_DROP` when it is optional and matched nothing.

    A trailing ``?`` marks a token the request can do without — an end frame, say,
    which the prevailing interface takes as an optional field and which a request
    carrying no second reference must simply not send. Without it there is no way
    to express an absent field: the token would raise, and writing the key with a
    ``null`` says something different to most backends than omitting it does.

    Absent the ``?`` an unresolved token still raises, which is what keeps a typo
    in a mapping from quietly becoming a dropped parameter.
    """
    token = token.strip()
    optional = token.endswith("?")
    if optional:
        token = token[:-1].strip()
    if token == "uuid":
        return str(uuid.uuid4())
    matches = parse(token).find(ctx)
    if not matches:
        if optional:
            return _DROP
        raise KeyError(f"unresolved template token: {{{token}}}")
    if _is_wildcard(token):
        # A wildcard names a list, and a list is what it renders to — one
        # prepare step run per listed reference hands the submit all its answers.
        return [m.value for m in matches]
    return matches[0].value


def parse_path(path: str):
    """A bare dot-path (``input.reference_images``) as a compiled expression."""
    return parse(path.strip())


def resolves(path: str, ctx: dict) -> bool:
    """Whether a bare dot-path (``input.image``) names anything in ``ctx`` — the
    test a conditional step is gated on. A present ``null`` names nothing."""
    return any(m.value is not None for m in parse(path.strip()).find(ctx))


def extract(jsonpath: str, body):
    """Evaluate a ``$.`` JSONPath against ``body``.

    Wildcard paths (``[*]`` / ``.*``) return the full list of matches; a plain
    path returns the single scalar (or ``None`` if it matches nothing).

    A path that continues *past a string* reads that string as JSON. Some task
    APIs answer with their result serialized into a string field, and a path that
    goes on below a string can match nothing any other way — so going on is the
    instruction to parse it. A path that stops at the string returns the string.
    """
    expr = parse(jsonpath)
    matches = [m.value for m in expr.find(body)]
    if not matches:
        matches = _find_through_strings(expr, body)
    if _is_wildcard(jsonpath):
        return matches
    return matches[0] if matches else None


def _find_through_strings(expr, body) -> list:
    """``expr`` evaluated one step at a time, parsing any string a step lands on
    before the next one descends. Only a plain chain of steps is walked; anything
    else (a recursive descent, a union) has already had its one ordinary try."""
    steps = []
    while isinstance(expr, Child):
        steps.append(expr.right)
        expr = expr.left
    steps.append(expr)
    values = [body]
    for step in reversed(steps):
        found = []
        for value in values:
            if isinstance(value, str) and value[:1] in ("{", "["):
                try:
                    value = json.loads(value)
                except ValueError:
                    continue
            found.extend(m.value for m in step.find(value))
        values = found
    return values


def _is_wildcard(jsonpath: str) -> bool:
    return "[*]" in jsonpath or ".*" in jsonpath
