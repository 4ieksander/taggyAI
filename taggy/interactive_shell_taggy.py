import click

from taggy_cli import cli


def interactive_shell():
    """Uruchamia interaktywną powłokę."""
    click.echo("Witaj w interaktywnej aplikacji z Click. Wpisz 'help', aby zobaczyć dostępne komendy.")
    while True:
        try:
            command = input(">").strip()
            if command:
                try:
                    cli.main(args=command.split())
                except:
                    click.echo("Nieprawidłowe polecenie.")
        except SystemExit:
            break
        except Exception as e:
            click.echo(f"Błąd: {e}")
            
if __name__ == "__main__":
    interactive_shell()