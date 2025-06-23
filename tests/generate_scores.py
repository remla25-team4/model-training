"""Script to run tests with coverage and extract ML and coverage scores."""

import re
import subprocess
import sys

def run_tests():
    """Run pytest with coverage and save output to file."""
    print("Running tests with coverage...\n")
    result = subprocess.run(
        ["coverage", "run", "-m", "pytest", "-v", "tests"],
        capture_output=True,
        text=True,
        check=False
    )

    with open("test_output.txt", "w", encoding="utf-8") as test_output:
        test_output.write(result.stdout)
        test_output.write(result.stderr)

    return result.returncode

def extract_scores():
    """Extract ML test score and coverage percentage from output."""
    with open("test_output.txt", encoding="utf-8") as test_output:
        output = test_output.read()

    ml_score_match = re.search(r"ML Test Score:\s*([0-9.]+)", output)
    ml_score = ml_score_match.group(1) if ml_score_match else "0.0"

    coverage_output = subprocess.check_output(["coverage", "report"], text=True)
    coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", coverage_output)
    coverage_score = coverage_match.group(1) if coverage_match else "0"

    return ml_score, coverage_score

if __name__ == "__main__":
    try:
        print("Running tests with coverage...\n")
        RETURN_CODE = run_tests()
        ML_SCORE, COVERAGE_SCORE = extract_scores()

        with open("test_scores.txt", "w", encoding="utf-8") as score_file:
            score_file.write(f"ML_SCORE={ML_SCORE}\n")
            score_file.write(f"COVERAGE_SCORE={COVERAGE_SCORE}\n")

        sys.exit(RETURN_CODE)

    except Exception as e:
        print("ERROR in generate_scores.py:", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


