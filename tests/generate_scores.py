"""Script to run tests with coverage and extract ML and coverage scores."""

import re
import subprocess
import sys
import traceback


def run_tests():
    """Run pytest with coverage and save output to file."""
    result = subprocess.run(
        ["coverage", "run", "--source=pipeline", "-m", "pytest", "-v", "tests"],
        capture_output=True,
        text=True,
        check=False
    )

    with open("test_output.txt", "w", encoding="utf-8") as output_file:
        output_file.write(result.stdout)
        output_file.write(result.stderr)

    return result.returncode

def extract_scores():
    """Extract ML test score and coverage percentage from output."""
    with open("test_output.txt", encoding="utf-8") as output_file:
        output = output_file.read()

    ml_score_match = re.search(r"ML Test Score:\s*(\d+)", output)
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

    except (subprocess.SubprocessError, OSError) as e:
        ERROR_MSG = f"ERROR in generate_scores.py: {e}"
        print(ERROR_MSG)
        TRACEBACK_STR = traceback.format_exc()

        # Save error to test_output.txt so GitHub can upload it
        with open("test_output.txt", "w", encoding="utf-8") as test_output:
            test_output.write(ERROR_MSG + "\n")
            test_output.write(TRACEBACK_STR)
        sys.exit(1)
