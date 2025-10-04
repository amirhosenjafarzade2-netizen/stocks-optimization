import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentiment import black_scholes_greeks, compute_sentiment, cluster_stocks
from optimizer import genetic_optimize_sentiment_weights, monte_carlo_simulation, optimize_portfolio

st.set_page_config(page_title="Options Greeks Sentiment Analyzer & Portfolio Optimizer", layout="wide")
st.title("Options Greeks Sentiment Analyzer & Portfolio Optimizer")
st.warning("Educational tool only. Not financial advice.")

if 'num_stocks' not in st.session_state:
    st.session_state.num_stocks = 2

with st.expander("Stock Configuration", expanded=True):
    num_stocks = st.number_input("Number of stocks", min_value=1, max_value=10, key="num_stocks")
    stock_names = []
    greeks_list = []
    ivs = []
    spots = []
    strikes = []
    expiries = []
    option_types = []
    cols = st.columns(2)
    for i in range(num_stocks):
        with cols[i % 2]:
            name = st.text_input(f"Stock {i+1} Ticker", value=f"Stock {i+1}")
            stock_names.append(name)
            auto_greeks = st.checkbox(f"Auto-compute Greeks for {name}", value=False)
            if auto_greeks:
                spot = st.number_input(f"Spot Price", value=100.0)
                strike = st.number_input(f"Strike Price", value=100.0)
                expiry = st.number_input(f"Time to Expiry (years)", value=0.25)
                iv = st.number_input(f"Implied Volatility", value=0.2)
                opt_type = st.selectbox(f"Option Type", ["call", "put"])
                current_rate = st.number_input("Current Risk-Free Rate", value=0.03)  # Shared but per stock for simplicity
                greeks = black_scholes_greeks(spot, strike, expiry, current_rate, iv, opt_type)
            else:
                delta = st.number_input(f"Delta", value=0.5)
                gamma = st.number_input(f"Gamma", value=0.02)
                theta = st.number_input(f"Theta (daily)", value=-0.01)
                vega = st.number_input(f"Vega (per 1%)", value=0.1)
                rho = st.number_input(f"Rho (per 1%)", value=0.05)
                greeks = {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}
            greeks_list.append(greeks)

with st.expander("Macro and Horizon Inputs", expanded=True):
    current_rate = st.number_input("Current Interest Rate", value=0.03)
    predicted_rate = st.number_input("Predicted Interest Rate", value=0.035)
    inflation = st.number_input("Inflation Rate", value=0.02)
    horizon = st.number_input("Investment Horizon (months)", value=3) / 12  # To years

uploaded_file = st.file_uploader("Upload CSV for Batch Input", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Assume columns: Ticker, Delta, Gamma, Theta, Vega, Rho, etc.
    stock_names = df['Ticker'].tolist()
    greeks_list = df[['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']].to_dict('records')

if st.button("Analyze Sentiment and Optimize"):
    scores = []
    sentiments = []
    for greeks in greeks_list:
        score, sentiment = compute_sentiment(greeks, current_rate, predicted_rate, inflation)
        scores.append(score)
        sentiments.append(sentiment)
    
    df_results = pd.DataFrame({
        'Stock': stock_names,
        'Sentiment Score': scores,
        'Sentiment': sentiments
    })
    st.table(df_results)
    
    # Clustering
    features = np.array([list(g.values()) for g in greeks_list])
    labels, centers = cluster_stocks(features)
    df_results['Cluster'] = labels
    st.table(df_results)
    
    # Optimize weights with GA
    opt_weights = genetic_optimize_sentiment_weights(features, np.array(scores) > 0)  # Bullish as positive
    
    # Expected returns (sentiment-based)
    expected_returns = np.array(scores) * 0.1  # Scale to returns
    volatilities = np.ones(len(scores)) * 0.2  # Placeholder
    correlations = np.eye(len(scores))
    
    # MC Simulation
    sim_returns = monte_carlo_simulation(expected_returns, volatilities, correlations, horizon)
    risks = sim_returns.std(axis=0)
    
    # Portfolio Opt
    port_weights = optimize_portfolio(expected_returns, risks)
    st.table(pd.DataFrame({'Stock': stock_names, 'Allocation (%)': port_weights * 100}))
    
    # Visuals
    fig, ax = plt.subplots()
    sns.barplot(x='Stock', y='Sentiment Score', data=df_results, ax=ax)
    st.pyplot(fig)
