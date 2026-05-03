from eda import run_eda
from load_and_merge import build_merged_dataset


def main() -> None:
    df = build_merged_dataset()

    print("\nRunning Phase I EDA...")
    run_eda(df)


if __name__ == "__main__":
    main()