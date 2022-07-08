import nox
import argparse
import argparse
import getpass

class PasswordPromptAction(argparse.Action):
    def __init__(self,
             option_strings,
             dest=None,
             nargs=0,
             default=None,
             required=False,
             type=None,
             metavar=None,
             help=None):
        super(PasswordPromptAction, self).__init__(
             option_strings=option_strings,
             dest=dest,
             nargs=nargs,
             default=default,
             required=required,
             metavar=metavar,
             type=type,
             help=help)

    def __call__(self, parser, args, values, option_string=None):
        password = getpass.getpass()
        setattr(args, self.dest, password)

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

    parser.add_argument(
        "username",
        type=str,
        nargs=1,
        help="Username for git"
    )


    parser.add_argument(
        "useremail",
        type=str,
        nargs=1,
        help="useremail for git"
    )

    parser.add_argument(
        "gitpassword",
        type=str,
        action=PasswordPromptAction,
        nargs=1,
        help="gitpassword for git"
    )

    args: argparse.Namespace = parser.parse_args(args=session.posargs)
    version: str = args.version.pop()
    username: str = args.username.pop()
    useremail: str = args.username.pop()
    gitpassword: str = args.username.pop()


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
    session.run("bump2version",  '--allow-dirty',version)


    session.log("Pushing the new tag")
    session.run("git", "config","--global","user.email",useremail,external=True)
    session.run("git", "config","--global","user.name",username,external=True)
    session.run("git", "config","--global","user.password",gitpassword,external=True)
    session.run("git", "remote","set-url","origin",f"git@github.com:{username}/zoish.git",external=True)
    session.run("git", "branch","temp-branch",external=True)
    session.run("git", "checkout", 'main',external=True)
    session.run("git", "merge", 'temp-branch',external=True)
    session.run("git", "push", 'origin','main',external=True)
    session.run("git", "branch", '--delete','temp-branch',external=True)


    # session.run("git", "push", external=True)
    session.run("git", "push", "--tags", external=True)