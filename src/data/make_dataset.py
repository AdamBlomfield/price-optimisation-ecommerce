"""
Generates a synthetic e-commerce dataset for the price optimisation model.

This script simulates daily sales data for a number of products over a specified
date range. It models several key real-world factors:
- Base price and demand for each product.
- Price elasticity: how changes in price affect demand.
- Seasonality: cyclical demand fluctuations throughout the year.
- Marketing spend impact on sales.
- Random noise to simulate unpredictable market variations.

The generated dataset is saved to the processed data directory, ready for
use in the feature engineering and model training stages.

Usage:
    python src/data/make_dataset.py
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to Python path to import config
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import project-specific configuration
import config


def generate_synthetic_data(params: dict) -> pd.DataFrame:
    """
    Core function to generate the synthetic dataset based on configuration.

    Args:
        params: A dictionary containing data generation parameters from the
                config file.

    Returns:
        A pandas DataFrame with the generated synthetic data.
    """
    logging.info("Starting synthetic data generation...")
    np.random.seed(42)  # for reproducibility

    # Unpack parameters from the config dictionary
    n_products = params["n_products"]
    start_date = params["start_date"]
    end_date = params["end_date"]

    # Generate date range and product SKUs
    dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date))
    product_ids = [f"SKU-{i}" for i in range(n_products)]
    logging.info(f"Generating data for {n_products} products from {start_date} to {end_date}.")

    # Assign stable characteristics to each product
    product_characteristics = {
        "base_price": np.random.normal(
            params["base_price_mean"], params["base_price_std"], n_products
        ),
        "base_demand": np.random.normal(
            params["base_demand_mean"], params["base_demand_std"], n_products
        ),
        "price_elasticity": np.random.normal(
            params["price_elasticity_mean"],
            params["price_elasticity_std"],
            n_products,
        ),
    }

    # Create the DataFrame structure
    df = pd.DataFrame(
        [(date, pid) for date in dates for pid in product_ids],
        columns=["date", "product_id"],
    )

    # Map product characteristics to the main DataFrame
    for i, pid in enumerate(product_ids):
        df.loc[df["product_id"] == pid, "base_price"] = product_characteristics[
            "base_price"
        ][i]
        df.loc[df["product_id"] == pid, "base_demand"] = product_characteristics[
            "base_demand"
        ][i]
        df.loc[
            df["product_id"] == pid, "price_elasticity"
        ] = product_characteristics["price_elasticity"][i]

    # --- Simulate Dynamic Variables ---

    # 1. Price: Simulate daily price fluctuations around the base price
    price_noise = np.random.normal(0, 0.05, len(df))
    df["price"] = df["base_price"] * (1 + price_noise)

    # 2. Seasonality: Model demand swings using a sine wave over the year
    df["day_of_year"] = df["date"].dt.dayofyear
    seasonality_effect = params["seasonality_amplitude"] * np.sin(
        2 * np.pi * (df["day_of_year"] - 80) / 365
    )

    # 3. Marketing Spend: Simulate marketing investment
    df["marketing_spend"] = np.random.gamma(2, 50, len(df))
    marketing_effect = params["marketing_effect_mean"] * np.log1p(
        df["marketing_spend"]
    )

    # --- Calculate Final Sales ---

    # Calculate the percentage change in price from the base price
    price_effect = df["price_elasticity"] * (
        (df["price"] - df["base_price"]) / df["base_price"]
    )

    # Combine all effects to simulate demand
    demand = df["base_demand"] * (
        1 + price_effect + seasonality_effect + marketing_effect
    )

    # Add random noise to the final sales figure
    noise = np.random.normal(0, params["noise_level"], len(df))
    df["sales"] = demand * (1 + noise)

    # Ensure sales and price are non-negative
    df["sales"] = df["sales"].clip(lower=0).round().astype(int)
    df["price"] = df["price"].clip(lower=0.01).round(2)
    df["marketing_spend"] = df["marketing_spend"].round(2)

    logging.info("Data generation complete. Finalising DataFrame.")

    # Select and reorder final columns
    final_cols = [
        "date",
        "product_id",
        "price",
        "sales",
        "marketing_spend",
        "day_of_year",
    ]
    return df[final_cols]


def main() -> None:
    """
    Main function to run the data generation pipeline.
    """
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate synthetic e-commerce data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.PROCESSED_DATA_DIR,
        help="Directory to save the output CSV file.",
    )
    args = parser.parse_args()

    # Generate the data
    synthetic_df = generate_synthetic_data(config.DATA_GENERATION)

    # Save the data to the specified location
    output_path = Path(args.output_dir) / config.DATA_GENERATION["output_filename"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved synthetic data to {output_path}")


if __name__ == "__main__":
    main()
