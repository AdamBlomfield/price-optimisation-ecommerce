"""
Project Configuration File

This file centralises all the configuration variables for the project,
including file paths, model parameters, and data generation settings.
Using a configuration file makes the code more modular, reproducible,
and easier to maintain.
"""
from pathlib import Path

# --- DIRECTORY PATHS ---

# The root directory of the project
PROJECT_DIR = Path(__file__).resolve().parents[0]

# Data-related directories
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
INTERIM_DATA_DIR = DATA_DIR / "02_interim"
PROCESSED_DATA_DIR = DATA_DIR / "03_processed"

# Other key directories
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_DIR / "logs"

# Ensure all necessary directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# --- LOGGING CONFIGURATION ---

# Logging settings for the project
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_file": LOGS_DIR / "price_optimisation.log",
    "max_file_size": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5,  # Keep 5 backup files
}


# --- DATA GENERATION PARAMETERS ---

# These parameters control the synthetic data generation process in src/data/make_dataset.py
DATA_GENERATION = {
    "n_products": 5,
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "base_price_mean": 50.0,
    "base_price_std": 10.0,
    "base_demand_mean": 100,
    "base_demand_std": 20,
    "price_elasticity_mean": -2.5,
    "price_elasticity_std": 0.5,
    "seasonality_amplitude": 0.3,
    "marketing_effect_mean": 0.1,
    "noise_level": 0.1,
    "output_filename": "synthetic_ecommerce_data.csv"
}


# --- MODEL TRAINING PARAMETERS ---

# These parameters are used by the model training script in src/models/train_model.py
MODEL_TRAINING = {
    # The target variable for the model
    "target_column": "sales",

    # Features to be used in the model
    "feature_columns": ["price", "marketing_spend", "day_of_year"],

    # Parameters for the Generalised Additive Model (GAM)
    "gam_params": {
        # Number of splines for non-linear features.
        # More splines can capture more complex patterns but risk overfitting.
        "n_splines_marketing": 10,
        "n_splines_seasonality": 20,
    },

    # Filename for the saved, trained model
    "model_output_filename": "price_elasticity_gam.pkl"
}


# --- DASHBOARD PARAMETERS ---

# Configuration for the Plotly Dash application in app.py
DASHBOARD = {
    "default_product_id": "SKU-0",
    "price_range_min_multiplier": 0.5, # e.g., 50% of average price
    "price_range_max_multiplier": 1.5, # e.g., 150% of average price
    "price_step": 0.50 # e.g., £0.50 increments on the slider
}
