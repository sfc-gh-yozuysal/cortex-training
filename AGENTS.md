# Agent Instructions for `cortex-training`

Always run the tests before and after changing any code in `src/cortex_training/`,
from the root of this repository:

```bash
uv pip install -e ".[dev]"
python3 -m pytest tests/ -q
ruff check
```

All tests must pass, and `ruff check` must report no errors.

## Context

`src/cortex_training/client/` is an HTTP client for the Cortex Training REST
API (`core.py`, with its authentication, telemetry and config normalizers
beside it), and `src/cortex_training/cli/` is the `cortex-training` (`ct`)
command line. This repository is the client only; the service implementation
lives elsewhere and is not public.

The reference documents in this repository are the contract to code against:

- `docs/reference/rest-api.md` — REST paths, request framing, polling, schemas.
  Treat it as the source of truth for anything a customer can observe, and
  update it in the same change as the client whenever the wire behavior moves.
- `docs/reference/cli.md` — commands, flags, configuration, environment vars.

## Public API docstrings

The docstrings of the public API are customer documentation. Each PyPI release's
docstrings are rendered verbatim into the Cortex Training API reference on
docs.snowflake.com, so a docstring change ships there only after the next release.
Never patch API wording in the docs repository: fix it here.

**Scope.** Every name in `cortex_training.__all__` except `wire` and `__version__`. For an
exported class, that's the class docstring and every public method, classmethod,
staticmethod, and property defined on that class, not inherited ones. The explicit
`__init__` of a class that isn't an exception counts too, since the reference renders
it. Private items (a leading underscore), and private parameters, are out of scope.

**Checks.** Two checks enforce these rules. Both are meant to be required to merge:
pytest already is, and the `lint (docstrings)` CI job is a check that will be required.

- **Ruff** enforces the structure: a docstring on every public item (`D101`, `D102`,
  `D103`, `D107`), the summary layout (`D205`, `D212`, `D415`), an `Args:` entry with a
  description for every parameter, `*args` and `**kwargs` included (`D417`), `Returns:`
  or `Yields:` when the function returns or yields a value (`DOC201`, `DOC402`), and
  `Raises:` for every exception the function raises with a `raise` statement (`DOC501`).
  It also checks the docstring form of the internal code in the same modules. Run it from
  the root of this repository, after `uv pip install -e ".[dev]"`:

  ```bash
  ruff check
  ```

  The rule set, the Ruff version, and the files it covers live in `pyproject.toml`
  (`[tool.ruff]`), so the command takes no arguments. It reads each module's `__all__` to
  tell public items from internal ones. When you export a name from a new module, add that
  module to `include` and the name to its `__all__`; the test fails until you do. Fix the
  docstring instead of adding `noqa` or an ignore on public code. Ruff can't see an
  exception that a helper or a library raises, and `DOC502` is off for that reason, so
  document those in `Raises:` too.
- **The test** (`tests/test_public_api_docstrings.py`) enforces what Ruff doesn't: S1 to
  S3 below in the parts Ruff leaves out, S4, and C1 to C7. It also checks that Ruff's scope
  still matches `cortex_training.__all__`.

**Rules:**

- **S1** The first line of every docstring is a one-sentence summary ending in a period
  (Ruff checks the layout and requires some punctuation). A deprecated item's summary
  reads `Deprecated: use <replacement>.` The test also requires a docstring on the items
  in modules whose names start with an underscore, because Ruff treats those as private.
- **S2** Google style: every parameter (except `self`/`cls`) appears in an `Args:`
  section with a description, and `*args` and `**kwargs` by name. Ruff checks the entries;
  the test requires the section. Deprecated items are exempt from the test, because the
  docs generator hides them.
- **S3** A method whose return annotation isn't `None` has a `Returns:` section, or
  `Yields:` for iterators. Properties need only the summary. The test checks it for every
  method, because Ruff skips one-line docstrings and summaries that start with "Return".
- **S4** Public signatures name nothing private: no default or annotation that uses a
  leading-underscore name. Make internal hooks private parameters (`_retry_on`), give
  constants a literal default, and type callbacks as `Callable[...]` or an exported type.
- **C1** No Latin abbreviations: `e.g.`, `i.e.`, `etc.`, `vs.`. Write "for example",
  "that is", "and so on", "versus".
- **C2** No repository paths or file names (`docs/`, `src/`, `tests/`, `recipes/`,
  `*.md`, `*.py`, "this repo"). Readers can't open them. Use a full https URL.
- **C3** No wire or REST internals: an HTTP method with a path, `/api/v2`, query strings,
  `DSSST1`, `application/octet-stream`, `proto3`. Also checked inside code spans.
- **C4** No private identifiers in prose: no backticked `_name`, no `:meth:` or
  `:class:` role that points at one.
- **C5** No implementation voice or notes: `we`, `our`, `TODO`, `FIXME`, `XXX`.
- **C6** No internal service terms: Neutrino, ZMD, zone manager, the zone, control
  plane. Also checked inside code spans.
- **C7** No em dashes. Use a colon or two sentences.

The C rules read the docstring prose without fenced code blocks and URLs, and (except
C3, C4, and C6) without inline code spans.

**Style** (not enforced, still expected):

- Describe observable behavior (inputs, outputs, errors, side effects), not mechanism
  such as chunking, retry rules, storage layout, or internal routes.
- Use contractions ("doesn't", "isn't"), the serial comma, and "or" instead of a slash.
- Give units for numbers ("300 to 604,800 seconds"). Use "might" or "can", not "may".
- Use American spelling ("canceled"), but keep a literal value such as `cancelled` in a
  code span.
- Keep the lead to a few short paragraphs. Name only things the reader can use:
  exported classes, public methods, and the exceptions a call raises.
- Use Markdown code spans (single backticks), not RST double backticks or roles.

**Baseline.** `tests/public_api_docstrings_baseline.txt` lists violations that are
accepted for now, one per line as `<qualname>: <rule> <detail>  # reason`. The test fails
on a violation that isn't listed, and on a listed one that no longer occurs, so remove
an entry when you fix it. The file is empty today: don't add to it without a reason.

## Troubleshooting a user's setup

When someone's `cortex-training` setup or connection fails (login, profiles,
which account or role is used, a 403, a refused privilege), read the agent
skill in `src/cortex_training/skills/cortex-training-setup/` first: `SKILL.md`,
then `troubleshooting.md` for specific errors.

## House rules

- **Never commit account identifiers, hostnames, PATs, request IDs, or job
  UUIDs**, including inside docstrings and example output. Use placeholders such
  as `ACCOUNT.snowflakecomputing.com` and `CORTEX_TRAINING_DB`.
- **Never reference non-public repositories, file paths, or internal codenames**
  in code comments, docstrings, or docs. A reader only has this repository.
- **No `print()` in library code.** The CLI writes machine-readable JSON to
  stdout; use `logger` so piping to `jq` keeps working.
- Binary payloads use the DSSST1 wire protocol in `src/cortex_training/wire.py`.
  Never use `torch.save`/pickle for a request body — the server rejects those
  frames.
- Assertions on serialized payloads should decode them (`wire.loads`) rather
  than only checking `isinstance(payload, bytes)`, which any encoding satisfies.
- When CLI commands, their output, or their error messages change, update the
  agent skill in `src/cortex_training/skills/` that describes them
  (`tests/test_skills.py` checks its commands still parse).
- When you change a default, a flag, or a field, update the docs that quote it in
  the same change: `docs/reference/cli.md`, `docs/reference/rest-api.md`, and the
  example READMEs plus `example.yaml`.
