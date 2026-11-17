# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merge gate for the public API docstrings.

The docstrings of every name in ``cortex_training.__all__`` are published verbatim on
docs.snowflake.com. The rules checked here are listed under "Public API docstrings" in
AGENTS.md. A violation is a string "<qualname>: <rule> <detail>"; the ones in
public_api_docstrings_baseline.txt are accepted for now.
"""

import ast
import inspect
import re
import textwrap
from pathlib import Path

import cortex_training

BASELINE_PATH = Path(__file__).with_name("public_api_docstrings_baseline.txt")
NOT_DOCUMENTED = {"wire", "__version__"}
GUIDE = "See the 'Public API docstrings' section of AGENTS.md for the rules."

_DEPRECATED = re.compile(r"Deprecated: use \w")
_SECTION = re.compile(r"^(Args|Returns|Yields|Raises|Example|Examples|Note|Notes|Attributes):\s*$")
_ENTRY = re.compile(r"^(\*{0,2}\w+)(?:\s*\(.*\))?:(?:\s+(.*))?$")
_URL = re.compile(r"https?://\S+")
_SPAN = re.compile(r"(`+)(?!`).+?(?<!`)\1", re.S)
_ROLE = re.compile(r":(?:meth|class|func|attr|mod|exc):`~?([^`]+)`")
_DOTTED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")

# rule -> (pattern, applied to text with inline code spans removed)
_PROSE_RULES = {
    "C1": (re.compile(r"\be\.g\b\.?|\bi\.e\b\.?|\betc\b\.?|\bvs\."), True),
    "C2": (
        re.compile(
            r"\b(?:docs|src|tests|recipes)/|\b[\w-]+\.(?:md|py)\b|\bthis repo(?:sitory)?\b",
            re.I,
        ),
        True,
    ),
    "C3": (
        re.compile(
            r"\b(?:GET|POST|PUT|PATCH|DELETE) /|/api/v2|\?\w+=|DSSST1|application/octet-stream|proto3"
        ),
        False,
    ),
    "C5": (re.compile(r"\b(?:[Ww]e|[Oo]ur)\b|\b(?:TODO|FIXME|XXX)\b"), True),
    "C6": (
        re.compile(r"\b(?:Neutrino|ZMD|zone manager|the zone|control plane)\b", re.I),
        False,
    ),
    "C7": (re.compile("\u2014"), True),
}


# ── Docstring parsing ───────────────────────────────────────────────────────


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _sections(doc: str) -> dict[str, list[str]]:
    """Map a Google-style section name to its body lines."""
    found: dict[str, list[str]] = {}
    current = None
    for line in doc.splitlines():
        match = _SECTION.match(line) if _indent(line) == 0 else None
        if match:
            current = found.setdefault(match.group(1), [])
        elif current is not None:
            if line.strip() and _indent(line) == 0:
                current = None
            else:
                current.append(line)
    return found


def _documented_args(doc: str) -> dict[str, str]:
    """Return {parameter name: description} from the Args section."""
    body = [line for line in _sections(doc).get("Args", []) if line.strip()]
    if not body:
        return {}
    entry_indent = _indent(body[0])
    args: dict[str, list[str]] = {}
    current = None
    for line in body:
        match = _ENTRY.match(line.strip()) if _indent(line) == entry_indent else None
        if match:
            current = args.setdefault(match.group(1).lstrip("*"), [])
            if match.group(2):
                current.append(match.group(2))
        elif current is not None:
            current.append(line.strip())
    return {name: " ".join(parts).strip() for name, parts in args.items()}


def _prose(doc: str) -> str:
    """The docstring without code blocks and URLs."""
    kept = []
    fenced = False
    literal_base = None
    for line in doc.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if literal_base is not None:
            if not line.strip() or _indent(line) > literal_base:
                continue
            literal_base = None
        if line.lstrip().startswith(">>>"):
            continue
        if line.rstrip().endswith("::"):
            literal_base = _indent(line)
        kept.append(line)
    return _URL.sub(" ", "\n".join(kept))


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _has_private_component(dotted: str) -> bool:
    return any(_is_private(part) for part in dotted.split("."))


# ── Rule checks ─────────────────────────────────────────────────────────────


def _check_prose(qualname: str, doc: str) -> list[str]:
    violations = []
    prose = _prose(doc)
    outside_spans = _SPAN.sub(" ", prose)
    for rule, (pattern, skip_spans) in _PROSE_RULES.items():
        text = outside_spans if skip_spans else prose
        seen = []
        for match in pattern.finditer(text):
            token = match.group(0).strip().rstrip(".") if rule == "C1" else match.group(0).strip()
            if token not in seen:
                seen.append(token)
                violations.append(f'{qualname}: {rule} "{token}"')
    for match in _SPAN.finditer(prose):
        span = match.group(0).strip("`")
        if any(_has_private_component(token) for token in _DOTTED.findall(span)):
            violations.append(f"{qualname}: C4 backticked private name `{span.strip()}`")
    for match in _ROLE.finditer(prose):
        if _has_private_component(match.group(1)):
            violations.append(f"{qualname}: C4 role pointing at private name `{match.group(1)}`")
    return violations


def _private_names(node: ast.AST) -> list[str]:
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and _is_private(child.id):
            names.append(child.id)
        elif isinstance(child, ast.Attribute) and _is_private(child.attr):
            names.append(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            try:
                names.extend(_private_names(ast.parse(child.value, mode="eval")))
            except SyntaxError:
                pass
    return names


def _check_signature_privacy(qualname: str, func) -> list[str]:
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError):
        return []
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    args = node.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    params = list(zip(positional, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
    violations = []
    for arg, default in params:
        if _is_private(arg.arg):
            continue
        if default is not None and _private_names(default):
            violations.append(f"{qualname}: S4 private default for {arg.arg}")
        if arg.annotation is not None and _private_names(arg.annotation):
            violations.append(f"{qualname}: S4 private annotation for {arg.arg}")
    for arg in (args.vararg, args.kwarg):
        if arg is not None and arg.annotation is not None and _private_names(arg.annotation):
            violations.append(f"{qualname}: S4 private annotation for {arg.arg}")
    if node.returns is not None and _private_names(node.returns):
        violations.append(f"{qualname}: S4 private annotation for return")
    return violations


def _check_summary(qualname: str, doc: str) -> list[str]:
    first = doc.splitlines()[0].strip() if doc.strip() else ""
    if not first:
        return [f"{qualname}: S1 missing docstring"]
    if not first.endswith(".") or re.search(r"[.!?] [A-Z]", first):
        return [f"{qualname}: S1 first line must be a one-sentence summary ending in a period"]
    if first.startswith("Deprecated") and not _DEPRECATED.match(first):
        return [f"{qualname}: S1 deprecated summary must read 'Deprecated: use <replacement>.'"]
    return []


def check_callable(qualname: str, func, *, kind: str = "function") -> list[str]:
    """Check one public function, method or property; kind is "function" or "property"."""
    target = func.fget if isinstance(func, property) else func
    doc = inspect.cleandoc(target.__doc__ or "")
    violations = _check_summary(qualname, doc)
    violations += _check_signature_privacy(qualname, target)
    violations += _check_prose(qualname, doc)
    if kind == "property":
        return violations
    deprecated = bool(_DEPRECATED.match(doc))
    signature = inspect.signature(target)
    if not deprecated:
        documented = _documented_args(doc)
        for name, param in signature.parameters.items():
            if name in ("self", "cls") or _is_private(name):
                continue
            if not documented.get(name):
                violations.append(f"{qualname}: S2 missing description for {name}")
    returns_none = signature.return_annotation in (None, "None")
    sections = _sections(doc)
    if (
        target.__name__ != "__init__"
        and not returns_none
        and not ({"Returns", "Yields"} & sections.keys())
    ):
        violations.append(f"{qualname}: S3 missing Returns or Yields section")
    return violations


def check_class(qualname: str, cls: type) -> list[str]:
    own_doc = cls.__dict__.get("__doc__") or ""
    if own_doc.startswith(f"{cls.__name__}("):  # dataclass-generated
        own_doc = ""
    violations = _check_summary(qualname, inspect.cleandoc(own_doc))
    violations += _check_prose(qualname, inspect.cleandoc(own_doc))
    is_exception = issubclass(cls, BaseException)
    for name, member in cls.__dict__.items():
        if name == "__init__":
            if inspect.isfunction(member) and member.__code__.co_filename != "<string>":
                if is_exception:
                    violations += _check_signature_privacy(f"{qualname}.__init__", member)
                else:
                    violations += check_callable(f"{qualname}.__init__", member)
            continue
        if _is_private(name) or name.startswith("__"):
            continue
        if isinstance(member, (classmethod, staticmethod)):
            member = member.__func__
        if isinstance(member, property):
            violations += check_callable(f"{qualname}.{name}", member, kind="property")
        elif inspect.isfunction(member):
            violations += check_callable(f"{qualname}.{name}", member)
    return violations


def collect_violations() -> list[str]:
    violations = []
    for name in cortex_training.__all__:
        if name in NOT_DOCUMENTED:
            continue
        obj = getattr(cortex_training, name)
        if inspect.isclass(obj):
            violations += check_class(name, obj)
        else:
            violations += check_callable(name, obj)
    return violations


# ── Baseline ────────────────────────────────────────────────────────────────


def load_baseline() -> dict[str, str]:
    """Return {violation: reason} from the baseline file."""
    entries = {}
    for number, line in enumerate(BASELINE_PATH.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, reason = line.partition("  # ")
        assert reason.strip(), f"baseline line {number} needs a trailing '  # reason'"
        entries[key.strip()] = reason.strip()
    return entries


# ── Tests ───────────────────────────────────────────────────────────────────


def test_public_api_docstrings_have_no_new_violations():
    new = sorted(set(collect_violations()) - set(load_baseline()))
    assert not new, f"{len(new)} public API docstring violation(s). {GUIDE}\n  " + "\n  ".join(new)


def test_public_api_docstring_baseline_has_no_stale_entries():
    stale = sorted(set(load_baseline()) - set(collect_violations()))
    assert not stale, (
        "These baseline entries no longer occur; remove them from "
        f"tests/public_api_docstrings_baseline.txt:\n  " + "\n  ".join(stale)
    )


def test_checker_flags_each_rule_family():
    def bad(self, a, retry_on=_private_hook, *args) -> int:
        """Does things, e.g. we call POST /x, see `_hidden` or src/a.py \u2014 TODO

        Args:
            a: The first.
        """

    violations = " | ".join(check_callable("bad", bad))
    for expected in (
        "S1",
        "S2 missing description for retry_on",
        "S2 missing description for args",
        "S3",
        "S4 private default for retry_on",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C7",
    ):
        assert expected in violations, (expected, violations)


def test_checker_accepts_a_documented_function_and_a_deprecated_one():
    def good(self, name: str, *, wait: float = 1.0) -> str:
        """Return the name.

        Args:
            name: The name, such as `x_`.
            wait: How long to wait, in seconds. A long description
                can continue on an indented line.

        Returns:
            The name.
        """

    def old(*args, **kwargs) -> str:
        """Deprecated: use good.

        Returns:
            The result.
        """

    assert check_callable("good", good) == []
    assert check_callable("old", old) == []


def _private_hook(exc):
    return False
