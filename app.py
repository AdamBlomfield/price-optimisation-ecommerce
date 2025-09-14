"""
E-commerce Price Optimisation Dashboard

This script launches a Plotly Dash application to visualise the results of the
price elasticity model. The dashboard allows users to select a product and
interactively explore how different price points are predicted to impact sales
and revenue, ultimately revealing the optimal price for revenue maximisation.

Usage:
    - Ensure the trained model and processed data exist.
    - Run the script from the command line: `python app.py`
    - Access the dashboard in a web browser at http://127.0.0.1:8050/
"""

import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Local application imports
import config


def load_artefacts(
    data_path: str, model_path: str
) -> tuple[pd.DataFrame, object]:
    """Loads the processed dataset and the trained model.

    Args:
        data_path: Path to the processed CSV data file.
        model_path: Path to the serialised model file (.pkl).

    Returns:
        A tuple containing the loaded DataFrame and the model object.
    """
    try:
        df = pd.read_csv(data_path)
        model = joblib.load(model_path)
        print("Successfully loaded data and model artefacts.")
        return df, model
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print(
            "Please ensure you have run the data and training pipeline first "
            "using 'make all' or 'make data' and 'make train'."
        )
        raise


def create_price_optimisation_figure(
    df_product: pd.DataFrame,
    model: object,
    selected_price: float,
) -> go.Figure:
    """
    Generates the price optimisation plot for a given product.

    Args:
        df_product: DataFrame filtered for a single product.
        model: The trained GAM model.
        selected_price: The price selected by the user on the slider.

    Returns:
        A Plotly graph object figure.
    """
    # 1. Generate a range of prices to predict on
    min_price = df_product["price"].min() * config.DASHBOARD[
        "price_range_min_multiplier"
    ]
    max_price = df_product["price"].max() * config.DASHBOARD[
        "price_range_max_multiplier"
    ]
    price_range = np.linspace(min_price, max_price, 200)

    # 2. Prepare features for prediction (using mean for non-price features)
    # This assumes the 'what-if' analysis is done holding other factors constant
    X_pred = pd.DataFrame(
        {
            "price": price_range,
            "marketing_spend": df_product["marketing_spend"].mean(),
            "day_of_year": df_product["day_of_year"].mean(),
        }
    )

    # 3. Predict sales and calculate revenue
    predicted_sales = model.predict(
        X_pred[config.MODEL_TRAINING["feature_columns"]]
    )
    # Ensure sales are not negative
    predicted_sales = np.maximum(0, predicted_sales)
    predicted_revenue = predicted_sales * price_range

    # 4. Find the optimal price
    optimal_idx = np.argmax(predicted_revenue)
    optimal_price = price_range[optimal_idx]
    max_revenue = predicted_revenue[optimal_idx]
    optimal_sales = predicted_sales[optimal_idx]

    # 5. Predict for the selected price
    selected_sales = model.predict(
        pd.DataFrame(
            {
                "price": [selected_price],
                "marketing_spend": [df_product["marketing_spend"].mean()],
                "day_of_year": [df_product["day_of_year"].mean()],
            }
        )
    )[0]
    selected_sales = max(0, selected_sales)
    selected_revenue = selected_price * selected_sales

    # 6. Create the figure
    fig = go.Figure()

    # Revenue curve
    fig.add_trace(
        go.Scatter(
            x=price_range,
            y=predicted_revenue,
            mode="lines",
            name="Predicted Revenue",
            line=dict(color="blue", width=2),
        )
    )

    # Optimal point
    fig.add_trace(
        go.Scatter(
            x=[optimal_price],
            y=[max_revenue],
            mode="markers",
            name="Optimal Price",
            marker=dict(color="red", size=12, symbol="star"),
        )
    )

    # Selected point
    fig.add_trace(
        go.Scatter(
            x=[selected_price],
            y=[selected_revenue],
            mode="markers",
            name="Selected Price",
            marker=dict(color="green", size=10, symbol="circle"),
        )
    )

    fig.update_layout(
        title=f"Revenue Optimisation Curve for {df_product['product_id'].iloc[0]}",
        xaxis_title="Price (£)",
        yaxis_title="Predicted Total Revenue (£)",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
    )
    return (
        fig,
        optimal_price,
        max_revenue,
        optimal_sales,
        selected_revenue,
        selected_sales,
    )


def build_results_table(
    current_price,
    current_revenue,
    current_sales,
    selected_price,
    selected_revenue,
    selected_sales,
    optimal_price,
    max_revenue,
    optimal_sales,
) -> dbc.Table:
    """Builds a Bootstrap table to display scenario results."""
    header = [html.Thead(html.Tr([html.Th("Scenario"), html.Th("Price"), html.Th("Predicted Sales"), html.Th("Predicted Revenue")]))]
    body = [
        html.Tbody(
            [
                html.Tr([html.Td("Current Average"), html.Td(f"£{current_price:.2f}"), html.Td(f"{current_sales:,.0f} units"), html.Td(f"£{current_revenue:,.0f}")]),
                html.Tr([html.Td("Selected Price"), html.Td(f"£{selected_price:.2f}"), html.Td(f"{selected_sales:,.0f} units"), html.Td(f"£{selected_revenue:,.0f}")]),
                html.Tr([html.Td("Optimal Price"), html.Td(f"£{optimal_price:.2f}"), html.Td(f"{optimal_sales:,.0f} units"), html.Td(f"£{max_revenue:,.0f}")], className="table-success"),
            ]
        )
    ]
    return dbc.Table(header + body, bordered=True, hover=True, striped=True)


# --- Main Application Logic ---

def main() -> None:
    """Main function to setup and run the Dash application."""

    # 1. Load data and model
    processed_data_path = (
        config.PROCESSED_DATA_DIR / config.DATA_GENERATION["output_filename"]
    )
    model_path = (
        config.MODELS_DIR / config.MODEL_TRAINING["model_output_filename"]
    )
    df, model = load_artefacts(processed_data_path, model_path)

    # 2. Initialise the Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "Price Optimisation Dashboard"

    # 3. Define the app layout
    app.layout = dbc.Container(
        [
            # Header
            html.H1("E-commerce Price Optimisation Dashboard", className="my-4"),
            html.P(
                "Use the controls below to select a product and simulate "
                "different price points."
            ),
            html.Hr(),
            # Controls and Graph
            dbc.Row(
                [
                    # Controls Column
                    dbc.Col(
                        [
                            html.H4("Controls"),
                            html.Label("Select Product SKU:"),
                            dcc.Dropdown(
                                id="product-dropdown",
                                options=[
                                    {"label": sku, "value": sku}
                                    for sku in df["product_id"].unique()
                                ],
                                value=config.DASHBOARD["default_product_id"],
                            ),
                            html.Br(),
                            html.Label("Select Price Point (£):"),
                            # The slider will be dynamically updated by the callback
                            dcc.Slider(id="price-slider"),
                            html.Div(id="slider-output-container", className="mt-2")

                        ],
                        md=4,
                        style={"backgroundColor": "#f8f9fa", "padding": "20px"},
                    ),
                    # Graph Column
                    dbc.Col(
                        [
                            dcc.Graph(id="revenue-optimisation-graph")
                        ],
                        md=8,
                    ),
                ],
                className="my-4",
            ),
            # Results Table
            dbc.Row([dbc.Col(id="results-table", md=12)]),
        ],
        fluid=True,
    )

    # 4. Define callbacks for interactivity
    @app.callback(
        Output("price-slider", "min"),
        Output("price-slider", "max"),
        Output("price-slider", "step"),
        Output("price-slider", "value"),
        Output("price-slider", "marks"),
        Input("product-dropdown", "value"),
    )
    def update_slider(product_id: str) -> tuple[float, float, float, float, dict]:
        """Update the price slider's range based on the selected product."""
        df_product = df[df["product_id"] == product_id]
        avg_price = df_product["price"].mean()
        min_p = round(avg_price * config.DASHBOARD["price_range_min_multiplier"])
        max_p = round(avg_price * config.DASHBOARD["price_range_max_multiplier"])
        step = config.DASHBOARD["price_step"]
        marks = {i: f"£{i}" for i in range(min_p, max_p + 1, int((max_p - min_p)/4))}
        return min_p, max_p, step, avg_price, marks


    @app.callback(
        Output("revenue-optimisation-graph", "figure"),
        Output("results-table", "children"),
        Output("slider-output-container", "children"),
        Input("product-dropdown", "value"),
        Input("price-slider", "value"),
    )
    def update_graph_and_table(
        product_id: str, selected_price: float
    ) -> tuple[go.Figure, dbc.Table, html.Div]:
        """
        Updates the main graph and results table when a new product
        or price is selected.
        """
        if not product_id or selected_price is None:
            return dash.no_update

        df_product = df[df["product_id"] == product_id]

        (
            fig,
            optimal_price,
            max_revenue,
            optimal_sales,
            selected_revenue,
            selected_sales,
        ) = create_price_optimisation_figure(df_product, model, selected_price)

        # Get current stats
        current_price = df_product["price"].mean()
        current_sales = df_product["sales"].mean()
        current_revenue = current_price * current_sales

        table = build_results_table(
            current_price, current_revenue, current_sales,
            selected_price, selected_revenue, selected_sales,
            optimal_price, max_revenue, optimal_sales
        )

        slider_output = f"Selected Price: £{selected_price:.2f}"

        return fig, table, slider_output

    # 5. Run the server
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
