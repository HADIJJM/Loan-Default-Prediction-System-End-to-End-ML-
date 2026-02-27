from src.train import train_pipeline
from src.predict import run_demo_predictions


def main():
    print("\n—————————————————————————————————————————")
    print(" LOAN DEFAULT PREDICTION SYSTEM - FULL RUN ")
    print("———————————————————————————————————————————")

    print(">> Step 1: Training Model...\n")
    train_pipeline()

    print("\n>> Step 2: Running Prediction Demo...\n")
    run_demo_predictions()

    print("\n>> Pipeline completed successfully.")


if __name__ == "__main__":
    main()