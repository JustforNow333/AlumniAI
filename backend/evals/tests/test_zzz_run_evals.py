"""Opt-in pytest entry point that runs the full eval suite after the unit tests.

This lives in the last-collected testpath (``backend/evals/tests``) with a
``zzz`` name so it is the final test pytest runs: the unit tests complete first,
then the GPT-5.4 (hybrid) evals run. It is skipped unless ``--run-evals`` is
passed, so the normal ``pytest`` run stays fast and key-free.

    python -m pytest -q                       # unit tests only (no evals)
    python -m pytest --run-evals              # unit tests, then hybrid evals
    python -m pytest --run-evals --eval-mode offline   # deterministic, no key
    python -m pytest --run-evals --eval-mode all       # unit + offline + hybrid

The eval mode(s) are parametrized in conftest.py via ``--eval-mode`` (repeatable;
``all`` expands to offline + hybrid), so each mode runs as its own test node.
"""

import os

import pytest


@pytest.mark.evals
def test_run_evals(request, eval_mode):
    if not request.config.getoption("--run-evals"):
        pytest.skip("eval suite is opt-in; pass --run-evals to run it")

    mode = eval_mode
    if mode != "offline" and not os.getenv("OPENAI_API_KEY"):
        pytest.skip(f"--eval-mode {mode} needs OPENAI_API_KEY in backend/.env")

    from evals.run_evals import main

    exit_code = main(["--mode", mode, "--fail-on-failures"])
    assert exit_code == 0, (
        f"eval suite reported failures in --eval-mode {mode}; "
        "see the JSON/Markdown report paths printed above"
    )
