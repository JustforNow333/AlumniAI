"""Pytest configuration shared across the backend test and eval suites.

Adds an opt-in ``--run-evals`` flag so a single ``pytest`` invocation can run
the normal unit tests first and then the GPT-5.4 (hybrid) eval suite last.
"""


def pytest_addoption(parser):
    group = parser.getgroup("evals")
    group.addoption(
        "--run-evals",
        action="store_true",
        default=False,
        help="After the unit tests, run the eval suite (see --eval-mode).",
    )
    group.addoption(
        "--eval-mode",
        action="append",
        default=None,
        help=(
            "Eval execution mode(s) used by --run-evals. Repeatable. Modes: "
            "offline (deterministic, no AI), hybrid (live GPT-5.4), "
            "classifier-live, smoke-live, or 'all' for offline + hybrid. "
            "Default: hybrid."
        ),
    )


def resolve_eval_modes(config):
    """Return the ordered, de-duplicated list of eval modes to run."""
    raw = config.getoption("--eval-mode") or ["hybrid"]
    modes = []
    for value in raw:
        expanded = ["offline", "hybrid"] if value == "all" else [value]
        for mode in expanded:
            if mode not in modes:
                modes.append(mode)
    return modes


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "evals: live eval-runner case; only runs when --run-evals is passed.",
    )


def pytest_generate_tests(metafunc):
    if "eval_mode" in metafunc.fixturenames:
        metafunc.parametrize("eval_mode", resolve_eval_modes(metafunc.config))
