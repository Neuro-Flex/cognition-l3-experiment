import os
import pytest
from pylint.lint import Run

def test_pylint():
    """Run pylint on the project."""
    results = Run(['models'], do_exit=False)
    score = results.linter.stats.global_note
    assert score >= 7.0, f"Code quality score {score} is below threshold"
