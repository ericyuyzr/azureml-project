import os
import argparse
import pandas as pd
import joblib


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )

    parser.add_argument(
        "--input-path",
        type=str,
        required=False,
        help="Path to input csv file"
    )

    args = parser.parse_args()


    print("Starting batch inference...")


    # 1. Load model
    print(f"Loading model from: {args.model_path}")

    model = joblib.load(args.model_path)

    print("Model loaded successfully")


    # 2. Load inference data
    if args.input_path:

        print(f"Reading input data from {args.input_path}")

        df = pd.read_csv(args.input_path)

    else:

        print("No input data provided. Using sample data.")

        df = pd.DataFrame(
            {
                "feature1": [10, 11, 12],
                "feature2": [14, 15, 16]
            }
        )


    print("Input data:")
    print(df)


    # Keep only features used during training
    X = df[["feature1", "feature2"]]


    # 3. Predict
    print("Running prediction...")

    predictions = model.predict(X)


    # 4. Save results
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)


    results = df.copy()
    results["prediction"] = predictions


    output_file = os.path.join(
        output_dir,
        "predictions.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )


    print(f"Predictions saved to {output_file}")

    print("Batch inference completed!")


if __name__ == "__main__":
    main()