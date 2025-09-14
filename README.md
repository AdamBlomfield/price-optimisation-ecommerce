# E-commerce Price Optimization for Revenue Maximization

## Overview

**This project demonstrates an end-to-end data science workflow to solve a critical business problem for a fictional e-commerce retailer, "Fashionista." The core objective is to move beyond a static, cost-plus pricing model to a dynamic, data-driven strategy.**

**By modeling the price elasticity of demand using a Generalized Additive Model (GAM), this project identifies optimal price points to maximize revenue. The final output is a fully interactive Plotly Dash dashboard that empowers business users to conduct "what-if" scenario analysis and make informed pricing decisions.**

### 🚀 Live Dashboard & Demo

**[INSERT LINK TO MY LIVE, DEPLOYED DASHBOARD HERE]**

**Below is a preview of the interactive dashboard, which allows for product selection and price-point simulation to see the direct impact on predicted revenue.**

**[INSERT A SCREENSHOT OR, IDEALLY, AN ANIMATED GIF OF THE DASHBOARD IN ACTION HERE]**

## The Business Problem

**Fashionista, a mid-sized online apparel retailer, faces significant challenges due to its static pricing strategy. This approach has led to:**

* **Margin Erosion:** Reactive discounting in response to competitor promotions without understanding the true impact on demand.
* **Lost Revenue Opportunities:** Underpricing premium and exclusive items by failing to capture the customer's true willingness to pay.
* **Inefficient Inventory Management:** Inability to adjust prices for seasonal items, leading to costly end-of-season markdowns or stock-outs of popular products.
* **Lack of Agility:** A failure to respond to dynamic market shifts, competitor actions, or changes in consumer demand.

The goal is to develop a tool that addresses these issues and provides a clear path to **increasing total revenue by up to 50%** **.**

## The Solution: A Data-Driven Approach

**This project tackles the problem by architecting a robust, reproducible data science solution.**

1. **High-Fidelity Synthetic Data:** To overcome the limitations of real-world transactional data (which often lacks price variation), a synthetic dataset was generated. This creates a controlled environment where the "ground truth" price elasticity is known, allowing for rigorous model validation.
2. **Interpretable Modeling with GAMs:** A simple linear regression would fail to capture the complex, non-linear effects of seasonality and marketing, leading to a biased estimate of price elasticity. Conversely, a "black-box" model like a Gradient Boosting Machine would be uninterpretable to business users. **Generalized Additive Models (GAMs)** were chosen as the ideal solution, as they:
   * **Provide a direct, interpretable coefficient for price elasticity (by modeling price as a linear term).**
   * **Flexibly capture complex, non-linear relationships of confounding variables (like seasonality and marketing spend) using smooth splines.**
3. **Interactive Decision-Support Dashboard:** The final model is deployed into an interactive Plotly Dash application. This tool is not a black box; it's a decision-support system that allows category managers to explore the trade-offs between price, sales volume, and total revenue.

## Key Features

* **Revenue Optimization Curve:** The core visualization that plots Predicted Revenue against Price, making the optimal price point intuitive and easy to identify.
* **Interactive "What-If" Analysis:** A price slider allows users to instantly see the predicted impact of any price change on sales and revenue.
* **Scenario Comparison Table:** A clear table that compares the "Current Strategy," the user's "Selected Strategy," and the model's "Optimal Strategy," quantifying the potential revenue lift of each decision.

## Tech Stack

* **Backend & Modeling:** Python, Pandas, NumPy, Scikit-learn, PyGAM
* **Dashboard:** Plotly Dash, Dash Bootstrap Components
* **Project Management:** Makefile, Cookiecutter Data Science structure
* **Testing:** Pytest
* **Deployment:** Gunicorn, Render (or other PaaS)

## Project Structure

**This repository follows the Cookiecutter Data Science project structure to ensure the work is organized, reproducible, and scalable.**

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── 01_raw           <- The original, immutable data dump.
│   ├── 02_interim       <- Intermediate data that has been transformed.
│   └── 03_processed     <- The final, canonical data sets for modeling.
├── models             <- Trained and serialized models.
├── notebooks          <- Jupyter notebooks for exploration and analysis.
├── reports
│   └── figures        <- Generated graphics and figures.
├── src                <- Source code for use in this project.
├── tests              <- Unit tests for the source code.
├── app.py             <- Main script for the Plotly Dash dashboard.
├── environment.yml    <- The requirements file for reproducing the analysis environment.
└── Makefile           <- Makefile with commands like `make data` or `make train`.

```

## Setup and Installation

**To run this project locally, follow these steps:**

1. **Clone the repository:**

   ```
   git clone [https://github.com/AdamBlomfield/price-optimisation-ecommerce.git](https://github.com/AdamBlomfield/price-optimisation-ecommerce.git)
   cd price-optimisation-ecommerce
   ```
2. **Create and activate the environment:**

   * **Using Conda:**
     ```
     conda env create -f environment.yml

     ```
3. **Create and activate the environment:**

   * **Using Conda:**
     ```
     conda env create -f environment.yml
     conda activate price-optimization-env
     ```
4. **Run the data and modeling pipeline:**
   The `<span class="selected">Makefile</span>` contains the commands to run the entire pipeline.

   ```
   # This will run the data generation script and train the model
   make all

   ```
   **Alternatively, you can run the steps individually:**

   ```
   make data      # Generates the synthetic dataset
   make train     # Trains the GAM model

   ```
5. **Launch the Dashboard:**

   ```
   python app.py

   ```
   The application will be available at `http://127.0.0.1:8050/`.
