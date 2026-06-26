from __future__ import annotations

from .train import main as train_main


def main() -> None:
    train_main(force_random_search=True)


if __name__ == "__main__":
    main()
