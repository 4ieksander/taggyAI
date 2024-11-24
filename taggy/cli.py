import click

@click.command()
@click.option('--tag', prompt='Enter your tag', help='The tag to add or manage')
def add_tag(tag):
    """Simple program that adds or manages a tag."""
    click.echo(f"Tag added: {tag}")

if __name__ == '__main__':
    add_tag()
