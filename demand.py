import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------
# Data loading
# ---------------------------------------------------

@st.cache_data
def load_data(csv_path: str = "Walmart.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Dates are in DD-MM-YYYY format in your file
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df


def build_store_df(df: pd.DataFrame, store_id: int, product_price: float) -> pd.DataFrame:
    """
    Convert Weekly_Sales to product units for a specific store.
    Assumes all sales are from the chosen product with given unit price.
    """
    df = df.copy()
    df["Units_Sold"] = df["Weekly_Sales"] / product_price
    store_df = df[df["Store"] == store_id][["Date", "Units_Sold"]].copy()
    store_df = store_df.rename(columns={"Date": "ds", "Units_Sold": "y"})
    store_df = store_df.sort_values("ds")
    return store_df


# ---------------------------------------------------
# Forecasting (Exponential Smoothing instead of Prophet)
# ---------------------------------------------------

def run_forecast(store_df: pd.DataFrame, horizon_weeks: int):
    """
    Fit an Exponential Smoothing model and return:
    - historical fitted values
    - future forecast values
    Both as dataframes with columns: ['ds', 'yhat'].
    """
    ts = store_df.set_index("ds")["y"].asfreq("W")
    ts = ts.sort_index()

    # Basic additive trend + seasonal model with yearly seasonality (52 weeks)
    model = ExponentialSmoothing(
        ts,
        trend="add",
        seasonal="add",
        seasonal_periods=52,
        initialization_method="estimated",
    ).fit()

    # Fitted (historical)
    fitted = model.fittedvalues
    hist_df = pd.DataFrame({"ds": fitted.index, "yhat": fitted.values})

    # Future forecast
    future_index = pd.date_range(
        start=ts.index[-1] + pd.Timedelta(weeks=1),
        periods=horizon_weeks,
        freq="W",
    )
    future_forecast = model.forecast(horizon_weeks)
    future_df = pd.DataFrame({"ds": future_index, "yhat": future_forecast.values})

    return hist_df, future_df


# ---------------------------------------------------
# Inventory calculations
# ---------------------------------------------------

def inventory_calculations(
    forecast_future: pd.DataFrame,
    horizon_weeks: int,
    service_level: float,
    lead_time_weeks: int,
    order_cost: float,
    holding_cost: float,
    num_sims: int,
):
    """
    Compute μ, σ from forecast horizon, then SS, ROP, EOQ and stockout probability.
    All in product UNITS.
    """
    mu = forecast_future["yhat"].mean()      # mean weekly demand (units)
    sigma = forecast_future["yhat"].std()    # std dev weekly demand (units)

    # Convert service level (e.g. 95) to Z using normal quantile
    Z = norm.ppf(service_level / 100.0)

    # Demand during lead time
    mu_L = mu * lead_time_weeks
    sigma_L = sigma * np.sqrt(lead_time_weeks)

    # Safety stock & ROP
    safety_stock = Z * sigma_L
    rop = mu_L + safety_stock

    # Economic Order Quantity (EOQ)
    annual_demand_units = mu * 52  # approximate
    EOQ = np.sqrt((2 * annual_demand_units * order_cost) / holding_cost)

    # Monte Carlo simulation of demand during lead time
    simulated_demand = np.random.normal(mu_L, sigma_L, num_sims)
    current_inventory = rop  # assume we hold inventory at ROP for risk calc
    stockout_prob = np.mean(simulated_demand > current_inventory)

    results = {
        "mu": mu,
        "sigma": sigma,
        "Z": Z,
        "mu_L": mu_L,
        "sigma_L": sigma_L,
        "safety_stock": safety_stock,
        "rop": rop,
        "EOQ": EOQ,
        "current_inventory": current_inventory,
        "stockout_prob": stockout_prob,
    }
    return results


# ---------------------------------------------------
# Streamlit App
# ---------------------------------------------------

st.set_page_config(page_title="Inventory Optimization Dashboard", layout="wide")

st.title("📦 Inventory Optimization & Risk Simulation")
st.write(
    """
    This app uses **Walmart weekly sales data** and assumes a single product
    (e.g., **bottled water**) with an average unit price to:
    - Forecast weekly demand using **Exponential Smoothing**
    - Compute **Safety Stock, Reorder Point (ROP), EOQ**
    - Estimate **stockout probability** via Monte Carlo simulation
    """
)

# Load data
df = load_data("Walmart.csv")
store_ids = sorted(df["Store"].unique())

# ---------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------
st.sidebar.header("⚙️ Configuration")

store_id = st.sidebar.selectbox("Store ID", store_ids, index=0)

product_name = st.sidebar.text_input("Product name", value="Bottled Water")
product_price = st.sidebar.number_input(
    "Average product price ($ per unit)", min_value=0.01, value=5.00, step=0.50
)

service_level = st.sidebar.slider(
    "Target service level (%)", min_value=80.0, max_value=99.9, value=95.0, step=0.1
)

lead_time_weeks = st.sidebar.number_input(
    "Lead time (weeks)", min_value=1, max_value=12, value=1, step=1
)

order_cost = st.sidebar.number_input(
    "Ordering cost per order ($)", min_value=1.0, value=300.0, step=10.0
)

holding_cost = st.sidebar.number_input(
    "Holding cost per unit per week ($)", min_value=0.001, value=0.05, step=0.01, format="%.3f"
)

forecast_horizon_weeks = st.sidebar.number_input(
    "Forecast horizon (weeks)", min_value=4, max_value=52, value=12, step=1
)

num_sims = st.sidebar.number_input(
    "Monte Carlo simulations", min_value=1000, max_value=100000, value=10000, step=1000
)

run_button = st.sidebar.button("🚀 Run Analysis")

# ---------------------------------------------------
# Main Logic
# ---------------------------------------------------
if not run_button:
    st.info("Set your parameters in the sidebar and click **🚀 Run Analysis**.")
else:
    # Build store-level demand in units
    store_df = build_store_df(df, store_id, product_price)

    if store_df.empty:
        st.error("No data for this store. Try another Store ID.")
    else:
        # Layout: two columns
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader(f"📈 Historical Weekly Demand — Store {store_id}")
            st.line_chart(store_df.set_index("ds")["y"])

        # Forecast
        hist_df, future_df = run_forecast(store_df, horizon_weeks=forecast_horizon_weeks)

        with col_left:
            st.subheader("🔮 Forecasted Weekly Demand (Exponential Smoothing)")
            plot_df = pd.concat([hist_df, future_df], ignore_index=True)
            plot_df = plot_df.set_index("ds")
            st.line_chart(plot_df["yhat"])

        # Inventory calculations
        results = inventory_calculations(
            forecast_future=future_df,
            horizon_weeks=forecast_horizon_weeks,
            service_level=service_level,
            lead_time_weeks=lead_time_weeks,
            order_cost=order_cost,
            holding_cost=holding_cost,
            num_sims=num_sims,
        )

        mu = results["mu"]
        sigma = results["sigma"]
        Z = results["Z"]
        mu_L = results["mu_L"]
        sigma_L = results["sigma_L"]
        safety_stock = results["safety_stock"]
        rop = results["rop"]
        EOQ = results["EOQ"]
        current_inventory = results["current_inventory"]
        stockout_prob = results["stockout_prob"]

        with col_right:
            st.subheader("📊 Inventory Optimization Summary")
            st.markdown(f"**Store ID:** `{store_id}`")
            st.markdown(f"**Product:** `{product_name}`")
            st.markdown(f"**Avg price:** `${product_price:.2f}` per unit")

            st.metric("Mean weekly demand (μ)", f"{mu:,.2f} units")
            st.metric("Std dev weekly demand (σ)", f"{sigma:,.2f} units")
            st.metric("Service level (Z)", f"{Z:.3f}")

            st.write("---")
            st.metric("Safety stock (SS)", f"{safety_stock:,.2f} units")
            st.metric("Reorder point (ROP)", f"{rop:,.2f} units")
            st.metric("EOQ", f"{EOQ:,.2f} units")

            st.write("---")
            st.metric("Current inventory (assumed = ROP)", f"{current_inventory:,.2f} units")
            st.metric("Stockout probability", f"{stockout_prob * 100:.2f}%")

        # Interpretation block (clean & friendly)
        st.subheader("🧾 Interpretation (Clear & Simple)")
        st.write(
            f"""
            Here’s what the numbers mean for **Store {store_id}** and **{product_name}** (≈ ${product_price:.2f}/unit):

            ### 🔹 Demand Forecast
            - The store typically sells **{mu:,.0f} units** per week.
            - Week-to-week demand fluctuates by about **{sigma:,.0f} units** on average.

            ### 🔹 Inventory Policy
            Based on your target service level of **{service_level:.1f}%** and a **{lead_time_weeks}-week lead time**:
            - You should keep about **{safety_stock:,.0f} units** as extra buffer stock (Safety Stock).
            - You should place a new order whenever inventory falls to around **{rop:,.0f} units** (Reorder Point).

            ### 🔹 Order Quantity (EOQ)
            - Given your ordering cost (${order_cost:,.2f}) and holding cost (${holding_cost:.3f}/unit/week),
              the most cost-efficient order size is about **{EOQ:,.0f} units**.
            - This balances:
                - Larger, less frequent orders (higher holding cost)
                - Smaller, more frequent orders (higher ordering cost)

            ### 🔹 Stockout Risk
            - We simulated **{num_sims:,}** possible demand scenarios during the lead time.
            - With this policy, the chance of a stockout while waiting for the next order is **{stockout_prob*100:.2f}%**.
            - This risk is consistent with your target service level.

            ---
            **Bottom line:**  
            These numbers help you decide *how much to keep on hand*, *when to reorder*, and *how much to order*  
            so that you avoid running out of stock while controlling inventory costs.
            """
        )
