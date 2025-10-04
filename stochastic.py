```python
import numpy as np
from optimizer import calculate_metrics
from sentiment import black_scholes_greeks, compute_sentiment

def simulate_scenarios(weights, expected_returns, return_stds, volatilities, vol_stds, correlations, sentiment_scores, sent_stds, inflation, tax_rates, risk_free_rate, current_rate, predicted_rate, asset_prices, times_to_maturity, strike_prices, implied_vols, option_types, use_sharpe, use_inflation, use_tax_rate, use_advanced_metrics, num_simulations, df=5):
    """
    Simulate portfolio scenarios using t-distributions, including dynamic Greeks.
    Parameters:
        weights: Portfolio weights
        expected_returns: Expected returns
        return_stds: Standard deviations for returns
        volatilities: Volatilities
        vol_stds: Standard deviations for volatilities
        correlations: Correlation matrix
        sentiment_scores: Initial sentiment scores
        sent_stds: Standard deviations for sentiment scores
        inflation: Inflation rate
        tax_rates: Tax rates per stock
        risk_free_rate: Risk-free rate
        current_rate: Current interest rate
        predicted_rate: Predicted interest rate
        asset_prices: Spot prices
        times_to_maturity: Option expiries
        strike_prices: Option strike prices
        implied_vols: Implied volatilities
        option_types: Option types ('call' or 'put')
        use_sharpe: Include Sharpe Ratio
        use_inflation: Include inflation
        use_tax_rate: Include tax rates
        use_advanced_metrics: Include VaR, Sortino
        num_simulations: Number of simulations
        df: t-distribution degrees of freedom
    Returns:
        Dictionary with metrics (mean, p5, p50, p95)
    """
    portfolio_returns = np.zeros(num_simulations)
    portfolio_vols = np.zeros(num_simulations)
    portfolio_sentiments = np.zeros(num_simulations)
    real_returns = np.zeros(num_simulations)
    after_tax_returns = np.zeros(num_simulations)
    sharpe_ratios = np.zeros(num_simulations) if use_sharpe else None
    var_values = np.zeros(num_simulations) if use_advanced_metrics else None
    sortino_ratios = np.zeros(num_simulations) if use_advanced_metrics else None

    num_stocks = len(weights)
    cov_matrix = np.diag(vol_stds) @ correlations @ np.diag(vol_stds)
    L = np.linalg.cholesky(cov_matrix)
    scale = np.sqrt((df - 2) / df)

    for i in range(num_simulations):
        t_samples = np.random.standard_t(df, size=(num_stocks, 3)) * scale
        sim_returns = expected_returns + return_stds * t_samples[:, 0]
        sim_vols = volatilities + (L @ t_samples[:, 1])
        sim_sent_scores = sentiment_scores + sent_stds * t_samples[:, 2]

        sim_vols = np.maximum(sim_vols, 1e-6)
        sim_cov_matrix = np.diag(sim_vols) @ correlations @ np.diag(sim_vols)

        # Simulate stock prices and update Greeks
        sim_prices = np.array([p * np.exp(r - 0.5 * v**2 + v * np.random.normal()) for p, r, v in zip(asset_prices or [100]*num_stocks, sim_returns, sim_vols)])
        sim_ttm = np.array([max(t - 1/252, 0.01) for t in times_to_maturity or [0.25]*num_stocks])
        sim_greeks = [
            black_scholes_greeks(
                S=p, K=k, T=t, r=current_rate, sigma=iv, option_type=ot
            ) if p is not None else {'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0, 'Rho': 0}
            for p, k, t, iv, ot in zip(sim_prices, strike_prices or [100]*num_stocks, sim_ttm, implied_vols or [0.2]*num_stocks, option_types or ['call']*num_stocks)
        ]
        sim_sent = np.array([compute_sentiment(g, current_rate, predicted_rate, inflation)[0] for g in sim_greeks]) + sim_sent_scores

        metrics = calculate_metrics(
            weights=weights,
            expected_returns=sim_returns,
            cov_matrix=sim_cov_matrix,
            sentiment_scores=sim_sent,
            inflation=inflation,
            tax_rate=tax_rates,
            risk_free_rate=risk_free_rate,
            use_sharpe=use_sharpe,
            use_inflation=use_inflation,
            use_tax_rate=use_tax_rate,
            use_advanced_metrics=use_advanced_metrics
        )

        portfolio_returns[i] = metrics['Portfolio Return']
        portfolio_vols[i] = metrics['Portfolio Volatility']
        portfolio_sentiments[i] = metrics['Portfolio Sentiment Score']
        real_returns[i] = metrics['Real Return'] if use_inflation else metrics['Portfolio Return']
        after_tax_returns[i] = metrics['After-Tax Return'] if use_tax_rate else metrics['Portfolio Return']
        if use_sharpe:
            sharpe_ratios[i] = metrics['Sharpe Ratio']
        if use_advanced_metrics:
            var_values[i] = metrics['VaR 5%']
            sortino_ratios[i] = metrics['Sortino Ratio']

    results = {
        'Portfolio Return': {
            'mean': np.mean(portfolio_returns),
            'p5': np.percentile(portfolio_returns, 5),
            'p50': np.percentile(portfolio_returns, 50),
            'p95': np.percentile(portfolio_returns, 95),
            'samples': portfolio_returns
        },
        'Portfolio Volatility': {
            'mean': np.mean(portfolio_vols),
            'p5': np.percentile(portfolio_vols, 5),
            'p50': np.percentile(portfolio_vols, 50),
            'p95': np.percentile(portfolio_vols, 95),
            'samples': portfolio_vols
        },
        'Portfolio Sentiment Score': {
            'mean': np.mean(portfolio_sentiments),
            'p5': np.percentile(portfolio_sentiments, 5),
            'p50': np.percentile(portfolio_sentiments, 50),
            'p95': np.percentile(portfolio_sentiments, 95),
            'samples': portfolio_sentiments
        }
    }
    if use_inflation:
        results['Real Return'] = {
            'mean': np.mean(real_returns),
            'p5': np.percentile(real_returns, 5),
            'p50': np.percentile(real_returns, 50),
            'p95': np.percentile(real_returns, 95),
            'samples': real_returns
        }
    if use_tax_rate:
        results['After-Tax Return'] = {
            'mean': np.mean(after_tax_returns),
            'p5': np.percentile(after_tax_returns, 5),
            'p50': np.percentile(after_tax_returns, 50),
            'p95': np.percentile(after_tax_returns, 95),
            'samples': after_tax_returns
        }
    if use_sharpe:
        results['Sharpe Ratio'] = {
            'mean': np.mean(sharpe_ratios),
            'p5': np.percentile(sharpe_ratios, 5),
            'p50': np.percentile(sharpe_ratios, 50),
            'p95': np.percentile(sharpe_ratios, 95),
            'samples': sharpe_ratios
        }
    if use_advanced_metrics:
        results['VaR 5%'] = {
            'mean': np.mean(var_values),
            'p5': np.percentile(var_values, 5),
            'p50': np.percentile(var_values, 50),
            'p95': np.percentile(var_values, 95),
            'samples': var_values
        }
        results['Sortino Ratio'] = {
            'mean': np.mean(sortino_ratios),
            'p5': np.percentile(sortino_ratios, 5),
            'p50': np.percentile(sortino_ratios, 50),
            'p95': np.percentile(sortino_ratios, 95),
            'samples': sortino_ratios
        }

    return results
