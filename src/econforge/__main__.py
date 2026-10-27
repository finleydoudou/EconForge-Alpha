"""Allow `python -m econforge` to use the project CLI."""

from econforge.cli import entrypoint


if __name__ == "__main__":
    entrypoint()

