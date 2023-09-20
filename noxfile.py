import nox

# Default sessions to run when no sessions are specified.
nox.options.sessions = ["tests_zoish", "lint_zoish"]

# List of test files to be executed.
test_files = [
    "tests/test_ShapPlotFeatures.py",
    "tests/test_shap_compared_with_other_selector_binery_classification_fasttreeshap.py",
    "tests/test_shap_compared_with_other_selector_binery_classification_shap.py",
    "tests/test_shap_compared_with_other_selector_multi_classification_fasttreeshap.py",
    "tests/test_shap_compared_with_other_selector_multi_classification_shap.py",
    "tests/test_shap_compared_with_other_selector_regression_fasttreeshap.py",
    "tests/test_shap_compared_with_other_selector_regression_shap.py",
    "tests/test_shap_feature_selector_with_n_feature_fasttreeshap.py",
    "tests/test_shap_feature_selector_with_n_feature_shap.py",
    "tests/test_shap_feature_selector_with_threshold_fasttreeshap.py",
    "tests/test_shap_feature_selector_with_threshold_shap.py",
    "tests/test_zoish.py"
]

@nox.session
def tests_zoish(session):
    """Install test dependencies and run pytest for all test files."""
    session.install("-r", "requirements_test.txt")
    
    for test_file in test_files:
        session.run("pytest", test_file)

@nox.session
def lint_zoish(session):
    """Run lint session using nox."""
    # Install linters.
    session.install("black", "isort")
    
    # Run isort and black.
    session.run("isort", "./zoish/")
    session.run("black", "./zoish/")