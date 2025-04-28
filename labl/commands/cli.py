import typer

app = typer.Typer(no_args_is_help=True)


@app.command(help="Get the version of the application")
def get_version():
    try:
        from labl import __version__ as version
    except ImportError:
        version = "unknown"
    typer.echo(f"labl version: {version}")


if __name__ == "__main__":
    app()
