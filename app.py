import streamlit as st
from optimizer import optimize_portfolio, genetic_optimize_sentiment_weights
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
from stochastic import simulate_scenarios
from multiperiod import multiperiod_simulation
from sentiment import black_scholes_greeks, compute_sentiment, cluster_stocks, interpret_clusters

st.set_page_config(page_title="Options Greeks Sentiment Analyzer & Portfolio Optimizer", layout="wide")

st.title("Options Greeks Sentiment Analyzer & Portfolio Optimizer")
st.markdown("Analyze stock sentiment using options Greeks and optimize portfolio allocation for maximum profit.")
st.warning("Educational purposes only. Not financial advice.")

# Initialize session state
if 'num_stocks' not in st.session_state:
    st.session_state.num_stocks = 2
if 'stock_names' not in st.session_state:
    st.session_state.stock_names = ["Stock 1", "Stock 2"]

# Save/Load/CSV inputs
col1, col2, col3 = st.columns(3)
with col1:
    serializable_state = {k: v for k, v in st.session_state.items() if k.startswith(('num_stocks', 'stock_names', 'ret_', 'vol_', 'tax_', 'corr_', 'price_', 'strike_', 'ttm_', 'ivol_', 'opt_type_', 'delta_', 'gamma_', 'theta_', 'vega_', 'rho_')) or k in ['current_rate', 'predicted_rate', 'inflation']}
    st.download_button("Save Inputs", data=json.dumps(serializable_state), file_name="sentiment_inputs.json")
with col2:
    uploaded_json = st.file_uploader("Load JSON Inputs", type="json")
    if uploaded_json:
        try:
            data = json.load(uploaded_json)
            valid_keys = set(st.session_state.keys()) | set(['num_stocks', 'stock_names'] + [f'ret_{i}' for i in range(10)] + [f'vol_{i}' for i in range(10)] + [f'tax_{i}' for i in range(10)] + [f'corr_{i}_{j}' for i in range(10) for j in range(i+1, 10)] + [f'price_{i}' for i in range(10)] + [f'strike_{i}' for i in range(10)] + [f'ttm_{i}' for i in range(10)] + [f'ivol_{i}' for i in range(10)] + [f'opt_type_{i}' for i in range(10)] + [f'delta_{i}' for i in range(10)] + [f'gamma_{i}' for i in range(10)] + [f'theta_{i}' for i in range(10)] + [f'vega_{i}' for i in range(10)] + [f'rho_{i}' for i in range(10)] + ['current_rate', 'predicted_rate', 'inflation'])
            for k, v in data.items():
                if k in valid_keys:
                    st.session_state[k] = v
            st.rerun()
        except Exception as e:
            st.error(f"Error loading JSON: {str(e)}")
with col3:
    uploaded_csv = st.file_uploader("Load CSV Inputs", type="csv")
    csv_data = None
    if uploaded_csv:
        try:
            csv_data = pd.read_csv(uploaded_csv)
            expected_cols = {'Ticker', 'Expected Return', 'Volatility', 'Tax Rate', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'Spot Price', 'Strike Price', 'Time to Maturity', 'Implied Volatility', 'Option Type'}
            if not all(col in csv_data.columns for col in ['Ticker', 'Expected Return', 'Volatility']):
                st.error("CSV must at least contain: Ticker, Expected Return, Volatility. Optional: Tax Rate, Delta, Gamma, Theta, Vega, Rho, Spot Price, Strike Price, Time to Maturity, Implied Volatility, Option Type")
                st.stop()
            if len(csv_data) > 10:
                st.error("CSV cannot contain more than 10 stocks.")
                st.stop()
            st.session_state.num_stocks = len(csv_data)
            st.session_state.stock_names = csv_data['Ticker'].tolist()
            for i, row in csv_data.iterrows():
                st.session_state[f'ret_{i}'] = row['Expected Return']
                st.session_state[f'vol_{i}'] = row['Volatility']
                st.session_state[f'tax_{i}'] = row.get('Tax Rate', 0.20)
                st.session_state[f'delta_{i}'] = row.get('Delta', 0.5)
                st.session_state[f'gamma_{i}'] = row.get('Gamma', 0.02)
                st.session_state[f'theta_{i}'] = row.get('Theta', -0.01)
                st.session_state[f'vega_{i}'] = row.get('Vega', 0.1)
                st.session_state[f'rho_{i}'] = row.get('Rho', 0.05)
                st.session_state[f'price_{i}'] = row.get('Spot Price', 100.0)
                st.session_state[f'strike_{i}'] = row.get('Strike Price', 100.0)
                st.session_state[f'ttm_{i}'] = row.get('Time to Maturity', 0.25)
                st.session_state[f'ivol_{i}'] = row.get('Implied Volatility', 0.2)
                st.session_state[f'opt_type_{i}'] = row.get('Option Type', 'call')
            st.rerun()
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            st.stop()

if st.button("Reset All Inputs"):
    st.session_state.clear()
    st.session_state.num_stocks = 2
    st.session_state.stock_names = ["Stock 1", "Stock 2"]
    st.rerun()

# Stock Configuration
with st.expander("Stock Configuration", expanded=True):
    num_stocks = st.number_input(
        "Enter number of stocks",
        min_value=1,
        max_value=10,
        key="num_stocks",
        help="Select the number of stocks (1-10)."
    )

    if num_stocks != len(st.session_state.stock_names):
        st.session_state.stock_names = [f"Stock {i+1}" for i in range(num_stocks)]

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

# Optimization method
with st.expander("Optimization Method", expanded=True):
    methods = ["Monte Carlo", "Genetic Algorithm", "Gradient Descent (Mean-Variance)", "SciPy (Constrained)"]
    method = st.selectbox(
        "Select optimization method",
        methods,
        help="Choose method for portfolio optimization."
    )

# Constraints
with st.expander("Constraints", expanded=False):
    use_constraints = st.checkbox("Use weight constraints", value=False)
    if use_constraints:
        min_weight = st.number_input(
            "Minimum weight per stock",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            help="Minimum allocation per stock (0 to 0.5)."
        )
        max_weight = st.number_input(
            "Maximum weight per stock",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            help="Maximum allocation per stock (0 to 1)."
        )
        if min_weight > max_weight:
            st.error("Minimum weight cannot exceed maximum weight.")
            st.stop()
    else:
        min_weight = max_weight = None

# Metrics configuration
with st.expander("Metrics Configuration", expanded=True):
    use_inflation = st.checkbox("Use Inflation", value=True)
    inflation = st.number_input(
        "Inflation rate",
        min_value=0.0,
        max_value=0.2,
        value=st.session_state.get('inflation', 0.03),
        key="inflation",
        help="Annual inflation rate."
    ) if use_inflation else 0.0

    use_tax_rate = st.checkbox("Use Tax Rate", value=True)

    use_sharpe = st.checkbox("Include Sharpe Ratio", value=True)
    risk_free_rate = st.number_input(
        "Risk-free rate",
        min_value=0.0,
        max_value=0.2,
        value=0.02,
        help="Annual risk-free rate."
    ) if use_sharpe else 0.0

    use_advanced_metrics = st.checkbox("Include Advanced Metrics (VaR, Sortino)", value=False)

    use_stochastic = st.checkbox("Stochastic Mode", value=False)
    if use_stochastic:
        num_simulations = st.number_input(
            "Number of simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            help="Number of Monte Carlo scenarios."
        )
        std_factor = st.number_input(
            "Uncertainty factor",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            help="Multiplies standard deviations for simulations."
        )

    use_multiperiod = st.checkbox("Multi-Period Mode", value=False)
    if use_multiperiod:
        horizon = st.number_input(
            "Investment horizon (years)",
            min_value=0.083,
            max_value=5.0,
            value=0.25,
            help="Default 0.25 = 3 months."
        )
        rebalance_freq = st.selectbox(
            "Rebalance frequency",
            ["Annual", "Quarterly", "Monthly"],
            help="How often to rebalance the portfolio."
        )
        num_mp_sim = st.number_input(
            "Number of multi-period simulations",
            min_value=100,
            max_value=5000,
            value=1000
        )

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
            exp_ret = st.number_input(
                f"Expected return",
                min_value=-1.0,
                max_value=1.0,
                value=st.session_state.get(f'ret_{i}', 0.10),
                key=f"ret_{i}",
                help="Annualized expected return."
            )
            vol = st.number_input(
                f"Volatility",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get(f'vol_{i}', 0.15),
                key=f"vol_{i}",
                help="Annualized volatility."
            )
            tax_rate = st.number_input(
                f"Tax rate",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get(f'tax_{i}', 0.20),
                key=f"tax_{i}",
                help="Tax rate for returns."
            ) if use_tax_rate else 0.0
            auto_greeks = st.checkbox(f"Auto-compute Greeks for {stock}", value=False)
            if auto_greeks:
                asset_price = st.number_input(
                    f"Spot Price",
                    min_value=0.0,
                    value=st.session_state.get(f'price_{i}', 100.0),
                    key=f"price_{i}",
                    help="Current stock price."
                )
                strike_price = st.number_input(
                    f"Strike Price",
                    min_value=0.0,
                    value=st.session_state.get(f'strike_{i}', 100.0),
                    key=f"strike_{i}",
                    help="Option strike price."
                )
                time_to_maturity = st.number_input(
                    f"Time to Maturity (years)",
                    min_value=0.01,
                    max_value=5.0,
                    value=st.session_state.get(f'ttm_{i}', 0.25),
                    key=f"ttm_{i}",
                    help="Time until option expiry."
                )
                implied_vol = st.number_input(
                    f"Implied Volatility",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get(f'ivol_{i}', 0.2),
                    key=f"ivol_{i}",
                    help="Option implied volatility."
                )
                opt_type = st.selectbox(
                    f"Option Type",
                    ["call", "put"],
                    index=0 if st.session_state.get(f'opt_type_{i}', 'call') == 'call' else 1,
                    key=f"opt_type_{i}",
                    help="Call or put option."
                )
                greeks = black_scholes_greeks(asset_price, strike_price, time_to_maturity, risk_free_rate, implied_vol, opt_type)
            else:
                delta = st.number_input(
                    f"Delta",
                    min_value=-1.0,
                    max_value=1.0,
                    value=st.session_state.get(f'delta_{i}', 0.5),
                    key=f"delta_{i}",
                    help="Option Delta."
                )
                gamma = st.number_input(
                    f"Gamma",
                    min_value=0.0,
                    value=st.session_state.get(f'gamma_{i}', 0.02),
                    key=f"gamma_{i}",
                    help="Option Gamma."
                )
                theta = st.number_input(
                    f"Theta (daily)",
                    value=st.session_state.get(f'theta_{i}', -0.01),
                    key=f"theta_{i}",
                    help="Daily Theta decay."
                )
                vega = st.number_input(
                    f"Vega (per 1%)",
                    min_value=0.0,
                    value=st.session_state.get(f'vega_{i}', 0.1),
                    key=f"vega_{i}",
                    help="Vega per 1% volatility change."
                )
                rho = st.number_input(
                    f"Rho (per 1%)",
                    value=st.session_state.get(f'rho_{i}', 0.05),
                    key=f"rho_{i}",
                    help="Rho per 1% rate change."
                )
                greeks = {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}
                asset_price = strike_price = time_to_maturity = implied_vol = None
                opt_type = 'call'
            expected_returns.append(exp_ret)
            volatilities.append(vol)
            tax_rates.append(tax_rate)
            asset_prices.append(asset_price)
            strike_prices.append(strike_price)
            times_to_maturity.append(time_to_maturity)
            implied_vols.append(implied_vol)
            option_types.append(opt_type)
            greeks_list.append(greeks)

# Macro inputs
with st.expander("Macro Inputs", expanded=True):
    current_rate = st.number_input(
        "Current Interest Rate",
        min_value=0.0,
        max_value=0.2,
        value=st.session_state.get('current_rate', 0.03),
        key="current_rate",
        help="Current annual risk-free rate."
    )
    predicted_rate = st.number_input(
        "Predicted Interest Rate",
        min_value=0.0,
        max_value=0.2,
        value=st.session_state.get('predicted_rate', 0.035),
        key="predicted_rate",
        help="Predicted annual risk-free rate."
    )

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
                        min_value=-1.0,
                        max_value=1.0,
                        value=st.session_state.get(f'corr_{i}_{j}', 0.0),
                        key=f"corr_{i}_{j}",
                        help="Correlation between stocks."
                    )
                    corr_df.iloc[i, j] = corr
                    corr_df.iloc[j, i] = corr
        correlations = corr_df.values

# Input format
with st.expander("Input Format", expanded=True):
    input_format = st.radio(
        "Input numbers as:",
        ("Decimal (e.g., 0.05)", "Percentage (e.g., 5)"),
        help="Choose how to input returns, volatilities, etc."
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
iterations = st.number_input(
    "Number of iterations",
    min_value=100,
    max_value=5000,
    value=1000,
    help="Number of iterations for optimization."
)

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
    ax.set_title("Correlation Matrix")
    return fig

@st.cache_data
def plot_efficient_frontier(portfolio_returns, portfolio_vols, opt_return, opt_vol):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(portfolio_vols, portfolio_returns, c='blue', alpha=0.3, s=10)
    ax.scatter(opt_vol, opt_return, c='red', marker='*', s=300, label='Optimal Portfolio')
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    ax.grid(True)
    ax.legend()
    return fig

@st.cache_data
def plot_histogram(data, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=30, color='blue', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    return fig

@st.cache_data
def plot_sentiment_bar(scores, stock_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=stock_names, y=scores, ax=ax)
    ax.set_title("Sentiment Scores")
    ax.set_xlabel("Stock")
    ax.set_ylabel("Sentiment Score")
    return fig

# Optimize button
if st.button("Analyze Sentiment & Optimize Portfolio"):
    progress_bar = st.progress(0)
    with st.spinner("Processing..."):
        try:
            # Validate inputs
            if np.any(np.array(volatilities) <= 0):
                st.error("Volatilities must be positive.")
                st.stop()
            if len(set(stock_names)) != len(stock_names):
                st.error("Stock names must be unique.")
                st.stop()
            if np.any(np.isnan(correlations)) or not np.allclose(correlations, correlations.T):
                st.error("Correlation matrix must be symmetric and valid.")
                st.stop()

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
            # Recompute scores with optimized weights
            sentiment_scores = [compute_sentiment(g, current_rate, predicted_rate, inflation, opt_sent_weights)[0] for g in greeks_list]
            progress_bar.progress(0.5)

            # Portfolio optimization
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

            # Stochastic simulation
            if use_stochastic:
                return_stds = np.array(expected_returns) * std_factor
                vol_stds = np.array(volatilities) * std_factor
                sent_stds = np.array(sentiment_scores) * std_factor
                stochastic_metrics = simulate_scenarios(
                    weights=weights,
                    expected_returns=np.array(expected_returns),
                    return_stds=return_stds,
                    volatilities=np.array(volatilities),
                    vol_stds=vol_stds,
                    correlations=correlations,
                    sentiment_scores=np.array(sentiment_scores),
                    sent_stds=sent_stds,
                    inflation=inflation,
                    tax_rates=np.array(tax_rates),
                    risk_free_rate=risk_free_rate,
                    current_rate=current_rate,
                    predicted_rate=predicted_rate,
                    asset_prices=asset_prices,
                    times_to_maturity=times_to_maturity,
                    strike_prices=strike_prices,
                    implied_vols=implied_vols,
                    option_types=option_types,
                    use_sharpe=use_sharpe,
                    use_inflation=use_inflation,
                    use_tax_rate=use_tax_rate,
                    use_advanced_metrics=use_advanced_metrics,
                    num_simulations=num_simulations
                )
                metrics.update(stochastic_metrics)
            progress_bar.progress(0.75)

            # Multi-period simulation
            if use_multiperiod:
                mp_metrics = multiperiod_simulation(
                    weights=weights,
                    expected_returns=np.array(expected_returns),
                    volatilities=np.array(volatilities),
                    correlations=correlations,
                    horizon=horizon,
                    rebalance_freq=rebalance_freq,
                    num_simulations=num_mp_sim
                )
                metrics.update(mp_metrics)
            progress_bar.progress(1.0)

        except Exception as e:
            st.error(f"Error during optimization: {str(e)}")
            st.stop()

    st.success("Optimization and Analysis Complete!")

    # Display sentiment weights
    with st.expander("Sentiment Weights", expanded=True):
        weights_df = pd.DataFrame({
            'Greek': ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
            'Weight': opt_sent_weights
        })
        st.table(weights_df)
        st.markdown("These weights are optimized to maximize separation between bullish and bearish stocks.")

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

    # Cluster results
    with st.expander("Stock Clusters", expanded=False):
        clusters, centers = cluster_stocks(features)
        cluster_labels = interpret_clusters(centers)
        sent_df['Cluster'] = [cluster_labels[c] for c in clusters]
        st.table(sent_df)
        st.markdown("Clusters group stocks by similar Greeks profiles (e.g., bullish high Delta, high volatility sensitivity).")

    # Portfolio metrics
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
