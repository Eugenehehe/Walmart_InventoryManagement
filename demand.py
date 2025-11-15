import math
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------
# Helper: approximate inverse normal CDF (no scipy)
# ---------------------------------------------------
def normal_ppf(p: float) -> float:
    """Approximate inverse CDF of standard normal using binary search."""
    def cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    lo, hi = -5.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------
# Load & prepare data
# ---------------------------------------------------
@st.cache_data
def load_data(csv_path: str = "Walmart.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df


def build_store_df(df: pd.DataFrame, store_id: int, product_price: float) -> pd.DataFrame:
    """Convert Weekly_Sales ($) -> Units_Sold for a given store."""
    df = df.copy()
    df["Units_Sold"] = df["Weekly_Sales"] / product_price
    store_df = df[df["Store"] == store_id][["Date", "Units_Sold"]].copy()
    store_df = store_df.rename(columns={"Date": "ds", "Units_Sold": "y"})
    store_df = store_df.sort_values("ds")
    return store_df


# ---------------------------------------------------
# Forecast model (simple moving average + trend)
# ---------------------------------------------------
def run_forecast(store_df: pd.DataFrame, horizon_weeks: int):
    """
    Simple forecasting using moving average + linear trend.
    No heavy libs so it runs everywhere.
    """
    ts = store_df.copy()
    ts = ts.set_index("ds")["y"]
    ts = ts.sort_index()

    # Choose a window for moving average
    WINDOW = 12 if len(ts) >= 12 else max(1, len(ts) // 3 or 1)

    ma = ts.rolling(WINDOW).mean().iloc[-1]
    if pd.isna(ma):
        ma = ts.mean()

    # Approximate trend as average change per week
    if len(ts) > WINDOW:
        trend = (ts.iloc[-1] - ts.iloc[-WINDOW]) / WINDOW
    elif len(ts) > 1:
        trend = (ts.iloc[-1] - ts.iloc[0]) / (len(ts) - 1)
    else:
        trend = 0.0

    future_dates = pd.date_range(ts.index[-1] + pd.Timedelta(weeks=1),
                                 periods=horizon_weeks, freq="W")

    future_vals = []
    for i in range(1, horizon_weeks + 1):
        future_vals.append(ma + trend * i)

    future_df = pd.DataFrame({"ds": future_dates, "yhat": future_vals})
    hist_df = pd.DataFrame({"ds": ts.index, "yhat": ts.values})

    return hist_df, future_df


# ---------------------------------------------------
# Inventory + scenario + cost calculations
# ---------------------------------------------------
def inventory_and_costs(
    future_df: pd.DataFrame,
    service_level: float,
    lead_time_weeks: int,
    order_cost: float,
    holding_cost: float,
    num_sims: int,
    demand_change_pct: float,
    leadtime_change_pct: float,
):
    """
    Compute:
    - μ, σ from forecast
    - Apply scenario adjustments
    - Safety Stock, ROP, EOQ
    - Stockout probability
    - Baseline vs optimized annual cost
    """

    # Base demand stats from forecast
    mu_raw = future_df["yhat"].mean()
    sigma_raw = future_df["yhat"].std()

    # Scenario: adjust demand and lead time
    demand_factor = demand_change_pct / 100.0  # e.g. 120 -> 1.2
    leadtime_factor = leadtime_change_pct / 100.0

    mu = mu_raw * demand_factor
    sigma = sigma_raw * demand_factor  # assume volatility scales roughly with demand

    eff_lead_time = lead_time_weeks * leadtime_factor

    # Convert service level (e.g. 95) to Z
    Z = normal_ppf(service_level / 100.0)

    # Demand during lead time
    mu_L = mu * eff_lead_time
    sigma_L = sigma * math.sqrt(eff_lead_time)

    # Optimized policy
    safety_stock = Z * sigma_L
    rop = mu_L + safety_stock
    annual_demand = mu * 52

    EOQ = math.sqrt((2 * annual_demand * order_cost) / holding_cost)

    # Baseline policy (naive): no safety stock, reorder when inventory = μ * L,
    # order quantity = μ * L (order exactly lead-time demand)
    baseline_Q = mu_L if mu_L > 0 else max(mu, 1.0)
    baseline_rop = mu_L
    baseline_safety_stock = 0.0

    # Monte Carlo simulation of demand during lead time for optimized policy
    simulated = np.random.normal(mu_L, sigma_L, num_sims)
    stockout_prob_opt = float((simulated > rop).mean())

    # Approximate annual ordering + holding cost:
    def annual_cost(Q, SS):
        if Q <= 0:
            return float("inf")
        avg_inventory = SS + Q / 2.0  # Safety stock + cycle stock
        ordering_cost_year = (annual_demand / Q) * order_cost
        holding_cost_year = avg_inventory * holding_cost * 52  # per week → per year
        return ordering_cost_year + holding_cost_year, avg_inventory

    baseline_cost, baseline_avg_inv = annual_cost(baseline_Q, baseline_safety_stock)
    optimized_cost, optimized_avg_inv = annual_cost(EOQ, safety_stock)

    savings = baseline_cost - optimized_cost

    # KPIs
    turnover_baseline = annual_demand / baseline_avg_inv if baseline_avg_inv > 0 else float("nan")
    turnover_opt = annual_demand / optimized_avg_inv if optimized_avg_inv > 0 else float("nan")
    effective_service_level = 1.0 - stockout_prob_opt

    results = {
        "mu": mu,
        "sigma": sigma,
        "Z": Z,
        "mu_L": mu_L,
        "sigma_L": sigma_L,
        "safety_stock": safety_stock,
        "rop": rop,
        "EOQ": EOQ,
        "stockout_prob": stockout_prob_opt,
        "annual_demand": annual_demand,
        "baseline_cost": baseline_cost,
        "optimized_cost": optimized_cost,
        "cost_savings": savings,
        "baseline_turnover": turnover_baseline,
        "optimized_turnover": turnover_opt,
        "baseline_rop": baseline_rop,
        "baseline_Q": baseline_Q,
        "effective_service_level": effective_service_level,
    }
    return results


# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(page_title="Inventory Optimization Dashboard", layout="wide")

st.title("📦 Inventory Optimization & Scenario Planner")

st.write(
    """
    This tool uses **Walmart weekly sales data** and assumes a single product
    (e.g., bottled water) to:
    - Forecast weekly demand (simple moving average + trend)
    - Compute **Safety Stock, Reorder Point (ROP), EOQ**
    - Compare **baseline vs optimized annual inventory cost**
    - Run **scenarios** (demand shocks, lead time changes)
    """
)

df = load_data("Walmart.csv")
store_ids = sorted(df["Store"].unique())

# Sidebar
st.sidebar.header("⚙️ Configuration")

store_id = st.sidebar.selectbox("Store", store_ids, index=0)

product_name = st.sidebar.text_input("Product name", value="Bottled Water")
product_price = st.sidebar.number_input("Product price ($/unit)", min_value=0.01, value=5.00, step=0.50)

service_level = st.sidebar.slider("Target service level (%)", 80.0, 99.9, 95.0, 0.1)
lead_time_weeks = st.sidebar.number_input("Lead time (weeks)", min_value=1, max_value=12, value=1, step=1)

order_cost = st.sidebar.number_input("Ordering cost per order ($)", min_value=1.0, value=300.0, step=10.0)
holding_cost = st.sidebar.number_input("Holding cost per unit per week ($)", min_value=0.001, value=0.05, step=0.01, format="%.3f")

forecast_horizon_weeks = st.sidebar.number_input("Forecast horizon (weeks)", min_value=4, max_value=52, value=12, step=1)
num_sims = st.sidebar.number_input("Monte Carlo samples", min_value=2000, max_value=50000, value=10000, step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Scenario Planning")

demand_change_pct = st.sidebar.number_input(
    "Demand scenario (% of current)", min_value=50.0, max_value=200.0, value=100.0, step=5.0,
    help="e.g., 120 = demand grows by 20%"
)

leadtime_change_pct = st.sidebar.number_input(
    "Lead time scenario (% of current)", min_value=50.0, max_value=300.0, value=100.0, step=10.0,
    help="e.g., 150 = lead time increases by 50%"
)

run_button = st.sidebar.button("🚀 Run Analysis")

# Main content
if not run_button:
    st.info("Set your parameters on the left and click **🚀 Run Analysis**.")
else:
    store_df = build_store_df(df, store_id, product_price)

    if store_df.empty:
        st.error("No data available for this store.")
    else:
        col_left, col_right = st.columns([2, 1])

        # Historical demand
        with col_left:
            st.subheader(f"📈 Historical Weekly Demand — Store {store_id}")
            st.line_chart(store_df.set_index("ds")["y"])

        # Forecast
        hist_df, future_df = run_forecast(store_df, forecast_horizon_weeks)

        with col_left:
            st.subheader("🔮 Forecasted Weekly Demand (MA + Trend)")
            combined = pd.concat([hist_df, future_df]).set_index("ds")
            st.line_chart(combined["yhat"])

        # Inventory & costs
        results = inventory_and_costs(
            future_df=future_df,
            service_level=service_level,
            lead_time_weeks=lead_time_weeks,
            order_cost=order_cost,
            holding_cost=holding_cost,
            num_sims=num_sims,
            demand_change_pct=demand_change_pct,
            leadtime_change_pct=leadtime_change_pct,
        )

        mu = results["mu"]
        sigma = results["sigma"]
        safety_stock = results["safety_stock"]
        rop = results["rop"]
        EOQ = results["EOQ"]
        stockout_prob = results["stockout_prob"]
        annual_demand = results["annual_demand"]
        baseline_cost = results["baseline_cost"]
        optimized_cost = results["optimized_cost"]
        cost_savings = results["cost_savings"]
        baseline_turnover = results["baseline_turnover"]
        optimized_turnover = results["optimized_turnover"]
        effective_service_level = results["effective_service_level"]

        with col_right:
            st.subheader("📊 Inventory Policy (Optimized)")
            st.markdown(f"**Store:** `{store_id}`")
            st.markdown(f"**Product:** `{product_name}`")
            st.markdown(f"**Price:** `${product_price:.2f}` per unit")

            st.metric("Mean weekly demand (μ)", f"{mu:,.2f} units")
            st.metric("Std dev (σ)", f"{sigma:,.2f} units")
            st.metric("Safety stock", f"{safety_stock:,.2f} units")
            st.metric("Reorder point (ROP)", f"{rop:,.2f} units")
            st.metric("EOQ", f"{EOQ:,.2f} units")
            st.metric("Stockout probability", f"{stockout_prob*100:.2f}%")
            st.metric("Effective service level", f"{effective_service_level*100:.2f}%")

        # Cost & KPI section
        st.subheader("💰 Annual Cost & KPIs")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Baseline policy** (no safety stock, order ≈ lead-time demand)")
            st.write(f"- Annual demand: **{annual_demand:,.0f} units**")
            st.write(f"- Approx. annual cost: **${baseline_cost:,.0f}**")
            st.write(f"- Inventory turnover: **{baseline_turnover:,.2f}x**")

        with col2:
            st.write("**Optimized policy** (EOQ + safety stock)")
            st.write(f"- Approx. annual cost: **${optimized_cost:,.0f}**")
            st.write(f"- Inventory turnover: **{optimized_turnover:,.2f}x**")

        with col3:
            st.write("**Impact**")
            st.write(f"- Estimated annual savings: **${cost_savings:,.0f}**")
            if cost_savings > 0:
                st.success("Your optimized policy is cheaper than the baseline.")
            else:
                st.warning("Your current settings don't reduce annual cost vs baseline—try adjusting parameters.")

        # Interpretation
        st.subheader("🧾 Executive Summary")

        st.write(
            f"""
            For **Store {store_id}** and product **{product_name}** (≈ ${product_price:.2f}/unit),
            under the selected scenario:

            - The model expects about **{mu:,.0f} units/week** on average, with a volatility of **{sigma:,.0f} units**.
            - With a **{service_level:.1f}%** target service level and an effective lead time of
              **{lead_time_weeks * (leadtime_change_pct / 100):.1f} weeks**, the recommended policy is:
              - Keep roughly **{safety_stock:,.0f} units** as safety stock.
              - Place a new order when inventory falls to **{rop:,.0f} units**.
              - Order about **{EOQ:,.0f} units** each time (EOQ).

            - Compared to a naive baseline policy (no safety stock, ordering roughly lead-time demand),
              the optimized policy is estimated to:
              - Change annual inventory cost from **${baseline_cost:,.0f}** to **${optimized_cost:,.0f}**.
              - Yield **estimated savings of ${cost_savings:,.0f} per year** at this store.
              - Improve inventory turnover from **{baseline_turnover:,.2f}x** to **{optimized_turnover:,.2f}x**.

            - Under the optimized policy, the simulated probability of a stockout during lead time
              is about **{stockout_prob*100:.2f}%**, corresponding to an effective service level of
              **{effective_service_level*100:.2f}%**.

            ---
            **How this is business-level:**  
            This dashboard lets planners test different demand and lead-time scenarios,
            compare baseline vs optimized policies, and see the financial and service-level
            impact of inventory decisions—exactly the type of analysis used in real retail
            and supply chain planning.
            """
        )
