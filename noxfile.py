import nox


@nox.session(python=[ "3.9.5"], venv_backend="venv")
def tests(session: nox.Session) -> None:
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.run("pytest")

@nox.session
def lint(session: nox.Session) -> None:
    session.install("flake8","black","isort")
    session.run("flake8","./src/scallyshap/feature_selector/")
    session.run("black","./src/")
    session.run("isort","./src/")
