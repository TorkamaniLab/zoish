import nox
import argparse

@nox.session(python=False)
def tests(session):
    session.run('poetry', 'shell')
    session.run('poetry', 'install')
    session.run('pytest')

@nox.session
def lint(session):
    session.install("flake8","black","isort")
    session.run("isort","./zoish/")
    session.run("black","./zoish/")
    session.run(
        'flake8',
        '--ignore=E501,I202,W503,E203',"./zoish/feature_selectors")

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
    args: argparse.Namespace = parser.parse_args(args=session.posargs)
    version: str = args.version.pop()

    # If we get here, we should be good to go
    # Let's do a final check for safety

    # TODO
    # I remove the below section to enforce version 
    # change and push without confirmation

    # confirm = input(
    #    f"You are about to bump the {version!r} version. Are you sure? [y/n]: "
    # )
    # Abort on anything other than 'y'
    # if confirm.lower().strip() != "y":
    #    session.error(f"You said no when prompted to bump the {version!r} version.")


    session.install("bump2version")

    session.log(f"Bumping the {version!r} version")
    session.run("bump2version", version)

    session.log("Pushing the new tag")
    session.run("git", "push", external=True)
    session.run("git", "push", "--tags", external=True)