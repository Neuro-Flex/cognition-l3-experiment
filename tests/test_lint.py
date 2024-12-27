import os
import sys
from io import StringIO
from pylint.lint import Run
import pytest

def test_pylint():
    """Run pylint on the project."""
    # Redirect stdout to capture pylint's output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Run pylint with exit=False
        results = Run(['models'], exit=False)
        score = results.linter.stats.global_note
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Check the code quality score
        assert score >= 7.0, f"Code quality score {score} is below threshold"
    except:
        # Restore stdout before raising any exception
        sys.stdout = old_stdout
        raise
