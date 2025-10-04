import numpy as np
import pygad  # For GA; if not available, fallback to scipy

def genetic_optimize_sentiment_weights(features, labels, generations=50):
    # Simple GA to optimize weights for sentiment scoring
    def fitness_func(ga_instance, solution, solution_idx):
        scores = np.dot(features, solution)
        # Fitness: separation between bullish/bearish (assume labels 1 bull, -1 bear)
        bull_mean = np.mean(scores[labels > 0])
        bear_mean = np.mean(scores[labels < 0])
        return abs(bull_mean - bear_mean)  # Maximize separation
    
    ga_instance = pygad.GA(num_generations=generations,
                           num_parents_mating=4,
                           fitness_func=fitness_func,
                           sol_per_pop=20,
                           num_genes=5,  # For 5 Greeks
                           init_range_low=-1.0,
                           init_range_high=1.0)
    ga_instance.run()
    return ga_instance.best_solution()[0]

def monte_carlo_simulation(expected_returns, volatilities, correlations, horizon, num_simulations=1000):
    # Simulate portfolio returns over horizon
    num_assets = len(expected_returns)
    cov_matrix = np.diag(volatilities) @ correlations @ np.diag(volatilities)
    returns = np.zeros((num_simulations, num_assets))
    for i in range(num_simulations):
        returns[i] = np.random.multivariate_normal(expected_returns / 252 * horizon * 252, cov_matrix * horizon, 1)  # Annual to horizon
    return returns

def optimize_portfolio(expected_returns, risks, method='GA', iterations=1000):
    # Simple mean-variance or GA for weights
    num_assets = len(expected_returns)
    if method == 'GA':
        def fitness(w):
            ret = np.dot(w, expected_returns)
            vol = np.sqrt(np.dot(w.T, np.dot(np.cov(risks.T), w)))
            return ret / vol if vol > 0 else 0
        # Use pygad similar to above...
        # Placeholder: equal weights
        weights = np.ones(num_assets) / num_assets
    else:
        weights = np.ones(num_assets) / num_assets
    return weights
