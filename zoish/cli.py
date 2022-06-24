import click


@click.command()
@click.option("--count", default=1, help="Number of greetings.")
@click.option("--name", prompt="Your name", help="The person to greet.")
def cli(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo(f"Hello {name}!")


@click.group()
def cligroup():
    pass


@cligroup.command()
@click.option(
    "--greeting",
    prompt="greeting from scally shap :)",
    help="Develop this for using command line!",
    default="Greeting from scally shap :)",
)
def greeting(greeting):
    click.echo(greeting)
