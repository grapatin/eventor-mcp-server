Copilot instructions

Generate concise, single‑file Python scripts.

Target Python 3.12; no backward‑compatibility boilerplate.

Use the uv package manager in all install or virtual‑env examples (e.g. uv pip install rich, uv venv .venv).

Prefer the Python standard library; suggest third‑party packages only when explicitly requested.

Keep error handling minimal: let exceptions propagate; use assert sparingly for sanity checks, avoid broad try/except blocks.

Skip type hints, docstrings, and extensive comments unless I ask for them.

Favour functional style (small functions) over classes; avoid OOP patterns unless needed.

Use modern language features that improve readability—f‑strings, pathlib.Path, and dataclasses—when they make code shorter.

Assume scripts run from the command line; include a simple if __name__ == "__main__": guard when appropriate.

Avoid adding logging frameworks, linters, CI configs, test scaffolding, or deployment scripts unless requested.

No GUI, asyncio, or multi‑threading code unless specified.

Keep suggestions playful and hacker‑friendly; skip enterprise patterns and compliance concerns.

