import math
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------
# Helper: approximate inverse normal CDF (no scipy)
# ---------------------------------------------------
def normal_ppf(p: float) -> float:
    """Approximate inverse CDF of standard normal using binary search."""
    def cdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
    lo, hi = -5, 5
    for _ in range(40):
        mid = (lo + hi) / 2
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ---------------------------------------------------
# Load data
# ---------------------------------------------------
@st.cache_data
def load_data(csv_path="Walmart.csv"):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

def build_store_df(df, store_id, product_price):
    df = df.copy()
    df["Units_Sold"] = df["Weekly_Sales"] / product_price
    store_df = df[df["Store"] == store_id][["Date", "Units_Sold"]]
    store_df = store_df.rename(columns={"Date": "ds", "Units_Sold": "y"})
    store_df = store_df.sort_values("ds")
    return store_df

# ---------------------------------------------------
# Forecast model (simple moving average + trend)
# ---------------------------------------------------
def run_forecast(store_df, horizon_weeks):
    ts = store_df.copy()
    ts = ts.set_index("ds")["y"]

    # Moving average window
    WINDOW = 12 if len(ts) >= 12 else max(1, len(ts)//3)

    ma = ts.rolling(WINDOW).mean().iloc[-1]
    if pd.isna(ma):
        ma = ts.mean()

    # Trend = average weekly change
    if len(ts) >= 2:
        trend = (ts.iloc[-1] - ts.iloc[-WINDOW]) / WINDOW if WINDOW < len(ts) else (ts.iloc[-1] - ts.iloc[0]) / len(ts)
    else:
        trend = 0

    # Forecast values
    future_dates = pd.date_range(ts.index[-1] + pd.Timedelta(weeks=1), periods=horizon_weeks, freq="W")

    future_vals = []
    for i in range(1, horizon_weeks + 1):
        future_vals.append(ma + trend * i)

    forecast_df = pd.DataFrame({"ds": future_dates, "yhat": future_vals})

    # Historical fitted = ts itself
    hist_df = pd.DataFrame({"ds": ts.index, "yhat": ts.values})

    return hist_df, forecast_df

# ---------------------------------------------------
# Inventory calculations
# ---------------------------------------------------
def inventory_calculations(future_df, service_level, lead_time_weeks, order_cost, holding_cost, num_sims):
    mu = future_df["yhat"].mean()
    sigma = future_df["yhat"].std()

    Z = normal_ppf(service_level / 100.0)

    mu_L = mu * lead_time_weeks
    sigma_L = sigma * math.sqrt(lead_time_weeks)

    safety_stock = Z * sigma_L
    rop = mu_L + safety_stock

    annual_demand = mu * 52
    EOQ = math.sqrt((2 * annual_demand * order_cost) / holding_cost)

    simulated = np.random.normal(mu_L, sigma_L, num_sims)
    stockout_prob = float((simulated > rop).mean())

    return mu, sigma, Z, safety_stock, rop, EOQ, stockout_prob

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("📦 Inventory Optimization (Deployable Version)")

df = load_data("Walmart.csv")
store_ids = sorted(df["Store"].unique())

# Sidebar inputs
store_id = st.sidebar.selectbox("Store", store_ids)
product_price = st.sidebar.number_input("Product price ($/unit)", value=5.0)
service_level = st.sidebar.slider("Service level (%)", 80.0, 99.9, 95.0)
lead_time_weeks = st.sidebar.number_input("Lead time (weeks)", value=1, min_value=1)
order_cost = st.sidebar.number_input("Ordering cost ($)", value=300.0)
holding_cost = st.sidebar.number_input("Holding cost ($ per unit per week)", value=0.05)
horizon_weeks = st.sidebar.number_input("Forecast horizon (weeks)", value=12, min_value=4, max_value=52)
num_sims = st.sidebar.number_input("Monte Carlo samples", value=10000, min_value=5000)

if st.sidebar.button("Run 🚀"):
    store_df = build_store_df(df, store_id, product_price)

    st.subheader("📈 Historical Demand (units)")
    st.line_chart(store_df.set_index("ds")["y"])

    hist_df, future_df = run_forecast(store_df, horizon_weeks)

    st.subheader("🔮 Forecast (Moving Average + Trend)")
    combined = pd.concat([hist_df, future_df]).set_index("ds")
    st.line_chart(combined["yhat"])

    mu, sigma, Z, SS, ROP, EOQ, stockout = inventory_calculations(
        future_df, service_level, lead_time_weeks, order_cost, holding_cost, num_sims
    )

    st.subheader("📊 Results")
    st.metric("Mean Weekly Demand (μ)", f"{mu:,.2f} units")
    st.metric("Std Dev Weekly Demand (σ)", f"{sigma:,.2f} units")
    st.metric("Safety Stock", f"{SS:,.2f} units")
    st.metric("Reorder Point (ROP)", f"{ROP:,.2f} units")
    st.metric("EOQ", f"{EOQ:,.2f} units")
    st.metric("Stockout Probability", f"{stockout*100:.2f}%")

    st.subheader("🧾 Interpretation")
    st.write(f"""
    - The store typically sells **{mu:,.0f} units/week**.
    - Demand fluctuates by **{sigma:,.0f} units**.
    - To meet a **{service_level:.1f}% service level**, you should:
        - Hold **{SS:,.0f} units** extra (safety stock)
        - Reorder when inventory hits **{ROP:,.0f} units**
    - The most cost-efficient order size is **{EOQ:,.0f} units**.
    - Stockout probability during lead time: **{stockout*100:.2f}%**
    """)

