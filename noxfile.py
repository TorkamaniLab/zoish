import argparse
import os
import nox

# Run tests with nox
@nox.session(python=False)
def tests(session):
    """
    Establishes poetry environment and runs pytest tests
    """
    session.run("poetry", "install", external=True)
    # Ensure the directory for pytest caching exists and is owned by the correct user
    os.makedirs('/tmp/.cache', exist_ok=True)
    os.chown('/tmp/.cache', os.getuid(), os.getgid())
    # Use a different directory for pytest caching
    session.env['XDG_CACHE_HOME'] = '/tmp/.cache'
    session.run("pytest", "tests/test_shap_feature_selector_with_n_feature.py")
    session.run("pytest", "tests/test_shap_feature_selector_with_threshold.py")

# Linting with nox
@nox.session
def lint(session):
    """
    Installs linting tools and run them against the codebase
    """
    session.install("flake8", "black", "isort")
    session.run("isort", "./zoish/")
    session.run("black", "./zoish/")
    session.run("flake8", "--ignore=E501,I202,W503,E203", "./zoish/")

# Release with nox
@nox.session
def release(session):
    """
    Kicks off an automated release process by creating and pushing a new tag.
    Invokes bump2version with the posarg setting the version.

    Usage:
    $ nox -s release -- [major|minor|patch]
    """
    parser = argparse.ArgumentParser(description="Release a semver version.")
    parser.add_argument(
        "version",
        type=str,
        nargs=1,
        help="The type of semver release to make.",
        choices={"major", "minor", "patch"},
    )
    parser.add_argument("username", type=str, nargs=1, help="Username for git")
    parser.add_argument("useremail", type=str, nargs=1, help="useremail for git")
    parser.add_argument("gitpassword", type=str, nargs=1, help="gitpassword for git")

    args = parser.parse_args(args=session.posargs)
    version = args.version.pop()
    username = args.username.pop()
    useremail = args.useremail.pop()
    gitpassword = args.gitpassword.pop()

    session.install("bump2version")

    session.log(f"Bumping the {version!r} version")
    session.run("bump2version", "--allow-dirty", version)

    session.log("Pushing the new tag")
    session.run("git", "config", "--global", "user.email", useremail, external=True)
    session.run("git", "config", "--global", "user.name", username, external=True)
    session.run(
        "git", "config", "--global", "user.password", gitpassword, external=True
    )
    session.run(
        "git",
        "remote",
        "set-url",
        "origin",
        f"https://{username}:{gitpassword}@github.com/TorkamaniLab/zoish.git",
        external=True,
    )
    session.run("git", "branch", "temp-branch", external=True)
    session.run("git", "checkout", "main", external=True)
    session.run("git", "merge", "temp-branch", external=True)
    session.run("git", "branch", "--delete", "temp-branch", external=True)
    session.run("git", "push", "origin", external=True)
    session.run("git", "push", "--tags", external=True)
