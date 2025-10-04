import numpy as np

def multiperiod_simulation(weights, expected_returns, volatilities, correlations, horizon, rebalance_freq, num_simulations):
    freq_map = {'Annual': 1, 'Quarterly': 4, 'Monthly': 12}
    periods_per_year = freq_map.get(rebalance_freq, 1)
    total_periods = int(horizon * periods_per_year)
    period_returns = expected_returns / periods_per_year
    period_vols = volatilities / np.sqrt(periods_per_year)
    period_cov = np.diag(period_vols) @ correlations @ np.diag(period_vols)
    
    final_wealths = []
    annualized_returns = []
    
    for _ in range(num_simulations):
        asset_values = np.array(weights)
        for p in range(total_periods):
            period_asset_returns = np.random.multivariate_normal(period_returns, period_cov)
            asset_values *= (1 + period_asset_returns)
            if (p + 1) % periods_per_year == 0:
                total = np.sum(asset_values)
                asset_values = total * weights
        final_wealth = np.sum(asset_values)
        ann_ret = (final_wealth ** (1 / horizon)) - 1 if final_wealth > 0 else 0
        final_wealths.append(final_wealth)
        annualized_returns.append(ann_ret)
    
    mp_metrics = {
        'Final Wealth': {
            'mean': np.mean(final_wealths),
            'p5': np.percentile(final_wealths, 5),
            'p50': np.median(final_wealths),
            'p95': np.percentile(final_wealths, 95),
            'samples': final_wealths
        },
        'Annualized Return': {
            'mean': np.mean(annualized_returns),
            'p5': np.percentile(annualized_returns, 5),
            'p50': np.median(annualized_returns),
            'p95': np.percentile(annualized_returns, 95)
        }
    }
    return mp_metrics
