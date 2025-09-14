"""
Trains a Generalised Additive Model (GAM) to model price elasticity.

This script takes the processed synthetic e-commerce data, defines a GAM
structure based on the project configuration, and trains the model to predict
sales based on price, marketing spend, and seasonality.

The trained model is then serialised and saved to the 'models' directory,
ready for use in the prediction pipeline or the interactive dashboard.

Usage:
    python src/models/train_model.py
"""
import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from pygam import LinearGAM, s

# Add project root to Python path to import config
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import project-specific configuration
import config


def train_model(df: pd.DataFrame, params: dict) -> LinearGAM:
    """
    Defines and trains the Generalised Additive Model.

    Args:
        df: The processed input DataFrame containing features and the target.
        params: A dictionary of model training parameters from the config.

    Returns:
        The trained LinearGAM model object.
    """
    logging.info("Starting model training...")

    # 1. Define features (X) and target (y)
    try:
        X = df[params["feature_columns"]]
        y = df[params["target_column"]]
        logging.info(f"Features used for training: {X.columns.tolist()}")
        logging.info(f"Target variable: {params['target_column']}")
    except KeyError as e:
        logging.error(f"Column not found in DataFrame: {e}")
        raise

    # 2. Build the GAM model structure
    # This creates a model of the form:
    # sales ~ linear(price) + spline(marketing_spend) + spline(day_of_year)
    gam_structure = s(
        X.columns.get_loc("marketing_spend"),
        n_splines=params["gam_params"]["n_splines_marketing"],
    ) + s(
        X.columns.get_loc("day_of_year"),
        n_splines=params["gam_params"]["n_splines_seasonality"],
    )

    # Note: `price` is treated as a linear term by default in LinearGAM
    # if it's not wrapped in a term like `s()` or `f()`.
    model = LinearGAM(gam_structure)

    logging.info("Fitting the GAM model...")
    model.fit(X, y)
    logging.info("Model training completed successfully.")

    return model


def main() -> None:
    """
    Main function to run the model training pipeline.
    """
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Train the price elasticity model.")
    parser.add_argument(
        "--input_path",
        type=str,
        default=config.PROCESSED_DATA_DIR
        / config.DATA_GENERATION["output_filename"],
        help="Path to the processed data file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.MODELS_DIR,
        help="Directory to save the trained model.",
    )
    args = parser.parse_args()

    # Load the processed data
    logging.info(f"Loading data from {args.input_path}...")
    try:
        input_df = pd.read_csv(args.input_path)
    except FileNotFoundError:
        logging.error(f"Data file not found at {args.input_path}. Please run 'make data' first.")
        return

    # Train the model
    trained_model = train_model(input_df, config.MODEL_TRAINING)

    # Save the trained model using joblib for efficiency
    output_path = (
        Path(args.output_dir) / config.MODEL_TRAINING["model_output_filename"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_model, output_path)
    logging.info(f"Successfully saved trained model to {output_path}")


if __name__ == "__main__":
    main()
