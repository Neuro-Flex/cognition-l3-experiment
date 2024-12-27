import os
import pytest
from pylint import epylint as lint

def test_lint():
    """Test that the codebase passes pylint checks."""
    # Get project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Directories to lint
    dirs_to_check = ['models', 'core', 'tests']
    
    for directory in dirs_to_check:
        path = os.path.join(root_dir, directory)
        if os.path.exists(path):
            (stdout, stderr) = lint.py_run(f'{path}', return_std=True)
            output = stdout.getvalue()
            errors = stderr.getvalue()
            
            # Print output for debugging
            print(output)
            if errors:
                print(errors)
            
            # Check if there are any errors/warnings
            assert "Your code has been rated at" in output
            rating = float(output.split("Your code has been rated at ")[1].split('/')[0])
            assert rating >= 8.0, f"Code quality rating {rating} is below threshold of 8.0"
