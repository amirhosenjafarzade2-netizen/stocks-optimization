import streamlit as st
from optimizer import optimize_portfolio, genetic_optimize_sentiment_weights
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
from stochastic import simulate_scenarios
from multiperiod import multiperiod_simulation
from sentiment import black_scholes_greeks, compute_sentiment, cluster_stocks

# Set page config
st.set_page_config(page_title="Options Greeks Sentiment Analyzer & Portfolio Optimizer", layout="wide")

# Initialize session state
if 'num_stocks' not in st.session_state:
    st.session_state.num_stocks = 2
if 'stock_names' not in st.session_state:
    st.session_state.stock_names = ["Stock 1", "Stock 2"]

# Title and description
st.title("Options Greeks Sentiment Analyzer & Portfolio Optimizer")
st.markdown("Analyze stock sentiment using options Greeks and optimize portfolio allocation.")
st.warning("Educational purposes only. Not financial advice.")

# Save/Load inputs
col1, col2 = st.columns(2)
with col1:
    serializable_state = {k: v for k, v in st.session_state.items() if k.startswith(('num_stocks', 'stock_names', 'ret_', 'vol_', 'sent_', 'tax_', 'corr_', 'price_', 'strike_', 'ttm_', 'ivol_', 'opt_type_')) or k in ['current_rate', 'predicted_rate']}
    st.download_button("Save Inputs", data=json.dumps(serializable_state), file_name="sentiment_inputs.json")
with col2:
    uploaded_file = st.file_uploader("Load Inputs", type="json")
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            valid_keys = set(st.session_state.keys()) | set(['num_stocks', 'stock_names'] + [f'ret_{i}' for i in range(10)] + [f'vol_{i}' for i in range(10)] + [f'sent_{i}' for i in range(10)] + [f'tax_{i}' for i in range(10)] + [f'corr_{i}_{j}' for i in range(10) for j in range(i+1, 10)] + [f'price_{i}' for i in range(10)] + [f'strike_{i}' for i in range(10)] + [f'ttm_{i}' for i in range(10)] + [f'ivol_{i}' for i in range(10)] + [f'opt_type_{i}' for i in range(10)] + ['current_rate', 'predicted_rate'])
            for k, v in data.items():
                if k in valid_keys:
                    st.session_state[k] = v
            st.rerun()
        except Exception as e:
            st.error(f"Error loading inputs: {str(e)}")

# Reset button
if st.button("Reset All Inputs"):
    st.session_state.clear()
    st.session_state.num_stocks = 2
    st.session_state.stock_names = ["Stock 1", "Stock 2"]
    st.rerun()

# Step 1: Stock Configuration
with st.expander("Stock Configuration", expanded=True):
    num_stocks = st.number_input(
        "Enter number of stocks",
        min_value=1,
        max_value=10,
        key="num_stocks",
        help="Select the number of stocks (1-10)."
    )

    # Update stock_names
    if num_stocks != len(st.session_state.stock_names):
        st.session_state.stock_names = [f"Stock {i+1}" for i in range(num_stocks)]

    # Stock names input
    st.subheader("Stock Names")
    cols = st.columns(2)
    stock_names = []
    for i in range(num_stocks):
        with cols[i % 2]:
            name = st.text_input(
                f"Stock {i+1} name",
                value=st.session_state.stock_names[i],
                key=f"stock_{i}"
            )
            stock_names.append(name)
    st.session_state.stock_names = stock_names

# Step 2: Optimization method
with st.expander("Optimization Method", expanded=True):
    methods = ["Monte Carlo", "Genetic Algorithm", "Gradient Descent (Mean-Variance)", "SciPy (Constrained)"]
    method = st.selectbox(
        "Select optimization method",
        methods
    )

# Constraints
with st.expander("Constraints", expanded=False):
    use_constraints = st.checkbox("Use weight constraints", value=False)
    if use_constraints:
        min_weight = st.number_input("Minimum weight per stock", min_value=0.0, max_value=0.5, value=0.0)
        max_weight = st.number_input("Maximum weight per stock", min_value=0.0, max_value=1.0, value=1.0)
    else:
        min_weight = max_weight = None

# Metrics configuration
with st.expander("Metrics Configuration", expanded=True):
    use_inflation = st.checkbox("Use Inflation", value=True)
    inflation = st.number_input("Inflation rate", value=0.03) if use_inflation else 0.0

    use_tax_rate = st.checkbox("Use Tax Rate", value=True)

    use_sharpe = st.checkbox("Include Sharpe Ratio", value=True)
    risk_free_rate = st.number_input("Risk-free rate", value=0.02) if use_sharpe else 0.0

    use_advanced_metrics = st.checkbox("Include Advanced Metrics (VaR, Sortino)", value=False)

    use_stochastic = st.checkbox("Stochastic Mode", value=False)
    if use_stochastic:
        num_simulations = st.number_input("Number of simulations", value=1000)
        std_factor = st.number_input("Uncertainty factor", value=0.2)

    use_multiperiod = st.checkbox("Multi-Period Mode", value=False)
    if use_multiperiod:
        horizon = st.number_input("Investment horizon (years)", value=0.25)  # Default 3 months
        rebalance_freq = st.selectbox("Rebalance frequency", ["Annual", "Quarterly", "Monthly"])
        num_mp_sim = st.number_input("Number of multi-period simulations", value=1000)

# Per stock metrics
with st.expander("Per Stock Metrics", expanded=True):
    st.subheader("Stock Parameters")
    expected_returns = []
    volatilities = []
    tax_rates = []
    asset_prices = []
    strike_prices = []
    times_to_maturity = []
    implied_vols = []
    option_types = []
    greeks_list = []
    cols = st.columns(2)
    for i, stock in enumerate(stock_names):
        with cols[i % 2]:
            st.markdown(f"**{stock}**")
            exp_ret = st.number_input(f"Expected return", value=0.10, key=f"ret_{i}")
            vol = st.number_input(f"Volatility", value=0.15, key=f"vol_{i}")
            tax_rate = st.number_input(f"Tax rate", value=0.20, key=f"tax_{i}") if use_tax_rate else 0.0
            auto_greeks = st.checkbox(f"Auto-compute Greeks for {stock}", value=False)
            if auto_greeks:
                asset_price = st.number_input(f"Spot Price", value=100.0, key=f"price_{i}")
                strike_price = st.number_input(f"Strike Price", value=100.0, key=f"strike_{i}")
                time_to_maturity = st.number_input(f"Time to Maturity (years)", value=0.25, key=f"ttm_{i}")
                implied_vol = st.number_input(f"Implied Volatility", value=0.2, key=f"ivol_{i}")
                opt_type = st.selectbox(f"Option Type", ["call", "put"], key=f"opt_type_{i}")
                greeks = black_scholes_greeks(asset_price, strike_price, time_to_maturity, risk_free_rate, implied_vol, opt_type)
            else:
                delta = st.number_input(f"Delta", value=0.5, key=f"delta_{i}")
                gamma = st.number_input(f"Gamma", value=0.02, key=f"gamma_{i}")
                theta = st.number_input(f"Theta", value=-0.01, key=f"theta_{i}")
                vega = st.number_input(f"Vega", value=0.1, key=f"vega_{i}")
                rho = st.number_input(f"Rho", value=0.05, key=f"rho_{i}")
                greeks = {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}
            expected_returns.append(exp_ret)
            volatilities.append(vol)
            tax_rates.append(tax_rate)
            asset_prices.append(asset_price if auto_greeks else None)
            strike_prices.append(strike_price if auto_greeks else None)
            times_to_maturity.append(time_to_maturity if auto_greeks else None)
            implied_vols.append(implied_vol if auto_greeks else None)
            option_types.append(opt_type if auto_greeks else "call")
            greeks_list.append(greeks)

# Macro inputs
with st.expander("Macro Inputs", expanded=True):
    current_rate = st.number_input("Current Interest Rate", value=0.03, key="current_rate")
    predicted_rate = st.number_input("Predicted Interest Rate", value=0.035, key="predicted_rate")

# Correlation matrix
with st.expander("Correlation Matrix", expanded=True):
    default_corr = st.checkbox("Use default correlation (0.2)", value=False)
    if default_corr:
        correlations = np.ones((num_stocks, num_stocks)) * 0.2
        np.fill_diagonal(correlations, 1.0)
    else:
        corr_df = pd.DataFrame(np.eye(num_stocks), index=stock_names, columns=stock_names)
        cols = st.columns(2)
        for i in range(num_stocks):
            for j in range(i+1, num_stocks):
                with cols[(i+j) % 2]:
                    corr = st.number_input(
                        f"Correlation: {stock_names[i]} - {stock_names[j]}",
                        value=0.0,
                        key=f"corr_{i}_{j}"
                    )
                    corr_df.iloc[i, j] = corr
                    corr_df.iloc[j, i] = corr
        correlations = corr_df.values

# Input format
with st.expander("Input Format", expanded=True):
    input_format = st.radio(
        "Input numbers as:",
        ("Decimal (e.g., 0.05)", "Percentage (e.g., 5)")
    )
    scale = 0.01 if input_format == "Percentage (e.g., 5)" else 1.0

# Apply scale
if scale == 0.01:
    expected_returns = [r * scale for r in expected_returns]
    volatilities = [v * scale for v in volatilities]
    tax_rates = [t * scale for t in tax_rates]
    inflation *= scale
    risk_free_rate *= scale
    implied_vols = [v * scale if v is not None else None for v in implied_vols]
    current_rate *= scale
    predicted_rate *= scale

# Iterations
iterations = st.number_input("Number of iterations", value=1000)

# Cache plots
@st.cache_data
def plot_pie(weights, stock_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(weights * 100, labels=stock_names, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax.set_title("Portfolio Allocation")
    return fig

@st.cache_data
def plot_heatmap(correlations, stock_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', ax=ax, xticklabels=stock_names, yticklabels=stock_names, vmin=-1, vmax=1, center=0)
    return fig

@st.cache_data
def plot_efficient_frontier(portfolio_returns, portfolio_vols, opt_return, opt_vol):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(portfolio_vols, portfolio_returns, c='blue', alpha=0.3, s=10)
    ax.scatter(opt_vol, opt_return, c='red', marker='*', s=300)
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    ax.grid(True)
    return fig

@st.cache_data
def plot_histogram(data, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=30, color='blue', alpha=0.7)
    ax.set_title(title)
    return fig

@st.cache_data
def plot_sentiment_bar(scores, stock_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=stock_names, y=scores, ax=ax)
    ax.set_title("Sentiment Scores")
    return fig

# Optimize button
if st.button("Analyze Sentiment & Optimize Portfolio"):
    progress_bar = st.progress(0)
    with st.spinner("Processing..."):
        try:
            # Compute sentiment
            sentiment_scores = []
            sentiments = []
            for greeks in greeks_list:
                score, sent = compute_sentiment(greeks, current_rate, predicted_rate, inflation)
                sentiment_scores.append(score)
                sentiments.append(sent)
            progress_bar.progress(0.25)

            # Optimize sentiment weights with GA
            features = np.array([[g['Delta'], g['Gamma'], g['Theta'], g['Vega'], g['Rho']] for g in greeks_list])
            opt_sent_weights = genetic_optimize_sentiment_weights(features, np.array(sentiment_scores))
            # Recompute scores with opt weights
            sentiment_scores = [compute_sentiment(g, current_rate, predicted_rate, inflation, opt_sent_weights)[0] for g in greeks_list]
            progress_bar.progress(0.5)

            # Portfolio opt, using sentiment as dividend analog
            weights, metrics = optimize_portfolio(
                method=method,
                expected_returns=np.array(expected_returns),
                volatilities=np.array(volatilities),
                correlations=correlations,
                sentiment_scores=np.array(sentiment_scores),
                inflation=inflation,
                tax_rate=np.array(tax_rates),
                risk_free_rate=risk_free_rate,
                iterations=iterations,
                use_sharpe=use_sharpe,
                use_inflation=use_inflation,
                use_tax_rate=use_tax_rate,
                use_advanced_metrics=use_advanced_metrics,
                min_weight=min_weight,
                max_weight=max_weight
            )

            # Stochastic
            if use_stochastic:
                return_stds = np.array(expected_returns) * std_factor
                vol_stds = np.array(volatilities) * std_factor
                sent_stds = np.array(sentiment_scores) * std_factor
                stochastic_metrics = simulate_scenarios(
                    weights, np.array(expected_returns), return_stds, np.array(volatilities), vol_stds, correlations,
                    np.array(sentiment_scores), sent_stds, inflation, np.array(tax_rates), risk_free_rate, current_rate, predicted_rate,
                    use_sharpe, use_inflation, use_tax_rate, use_advanced_metrics, num_simulations
                )
                metrics.update(stochastic_metrics)

            # Multi-period
            if use_multiperiod:
                mp_metrics = multiperiod_simulation(
                    weights, np.array(expected_returns), np.array(volatilities), correlations,
                    horizon, rebalance_freq, num_mp_sim
                )
                metrics.update(mp_metrics)

            progress_bar.progress(1.0)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

    st.success("Complete!")

    # Display allocation
    with st.expander("Optimized Portfolio Allocation", expanded=True):
        weights_df = pd.DataFrame({"Stock": stock_names, "Weight (%)": weights * 100})
        st.table(weights_df)
        fig_pie = plot_pie(weights, stock_names)
        st.pyplot(fig_pie)

    # Sentiment results
    with st.expander("Sentiment Analysis", expanded=True):
        sent_df = pd.DataFrame({"Stock": stock_names, "Sentiment Score": sentiment_scores, "Sentiment": sentiments})
        st.table(sent_df)
        fig_sent = plot_sentiment_bar(sentiment_scores, stock_names)
        st.pyplot(fig_sent)

    # Cluster
    with st.expander("Stock Clusters", expanded=False):
        clusters, _ = cluster_stocks(features)
        sent_df['Cluster'] = clusters
        st.table(sent_df)

    # Metrics
    with st.expander("Portfolio Metrics", expanded=True):
        normalized_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                normalized_metrics[k] = v
            else:
                normalized_metrics[k] = {'mean': v}
        display_data = {}
        for metric, values in normalized_metrics.items():
            display_data[metric] = {
                'Mean': values['mean'],
                'P5': values.get('p5', '-'),
                'P50': values.get('p50', '-'),
                'P95': values.get('p95', '-')
            }
        metrics_df = pd.DataFrame(display_data).T
        metrics_df['Mean'] = metrics_df['Mean'].apply(lambda x: f"{x:.4f}")
        for col in ['P5', 'P50', 'P95']:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        st.table(metrics_df)

    # Correlation heatmap
    with st.expander("Correlation Heatmap", expanded=False):
        fig_corr = plot_heatmap(correlations, stock_names)
        st.pyplot(fig_corr)

    # Efficient frontier
    if method in ["Monte Carlo", "Genetic Algorithm"]:
        with st.expander("Efficient Frontier", expanded=False):
            portfolio_returns = []
            portfolio_vols = []
            cov_matrix = np.diag(volatilities) @ correlations @ np.diag(volatilities)
            for _ in range(500):
                rand_weights = np.random.dirichlet(np.ones(num_stocks))
                port_ret = np.dot(rand_weights, expected_returns)
                port_vol = np.sqrt(np.dot(rand_weights.T, np.dot(cov_matrix, rand_weights)))
                portfolio_returns.append(port_ret)
                portfolio_vols.append(port_vol)
            fig_ef = plot_efficient_frontier(portfolio_returns, portfolio_vols, metrics.get('Portfolio Return', 0), metrics.get('Portfolio Volatility', 0))
            st.pyplot(fig_ef)

    # Stochastic distributions
    if use_stochastic:
        with st.expander("Stochastic Distributions", expanded=False):
            fig_ret = plot_histogram(normalized_metrics['Portfolio Return']['samples'], "Portfolio Returns Distribution")
            st.pyplot(fig_ret)
            fig_sent = plot_histogram(normalized_metrics['Portfolio Sentiment Score']['samples'], "Portfolio Sentiment Distribution")
            st.pyplot(fig_sent)

    # Multi-period
    if use_multiperiod:
        with st.expander("Multi-Period Simulation", expanded=False):
            fig_wealth = plot_histogram(normalized_metrics['Final Wealth']['samples'], "Final Wealth Distribution")
            st.pyplot(fig_wealth)
