import numpy as np
import pygad
from scipy import optimize as opt

def validate_correlation_matrix(corr):
    if not np.allclose(corr, corr.T):
        raise ValueError("Correlation matrix must be symmetric")
    eigenvalues = np.linalg.eigvals(corr)
    if np.any(eigenvalues < -1e-10):
        raise ValueError("Correlation matrix must be positive semi-definite")

def optimize_portfolio(method, expected_returns, volatilities, correlations, sentiment_scores, inflation, tax_rate, risk_free_rate, iterations, use_sharpe, use_inflation, use_tax_rate, use_advanced_metrics, min_weight=None, max_weight=None):
    num_stocks = len(expected_returns)
    
    # Validate inputs
    if np.any(volatilities <= 0):
        raise ValueError("Volatilities must be positive")
    validate_correlation_matrix(correlations)
    
    cov_matrix = np.diag(volatilities) @ correlations @ np.diag(volatilities)
    
    min_weights = [min_weight] * num_stocks if min_weight is not None else None
    max_weights = [max_weight] * num_stocks if max_weight is not None else None
    
    if method == "Monte Carlo":
        weights = monte_carlo_optimize(expected_returns, cov_matrix, iterations, num_stocks, risk_free_rate, min_weights, max_weights)
    elif method == "Genetic Algorithm":
        weights = genetic_algorithm_optimize(expected_returns, cov_matrix, iterations, num_stocks, risk_free_rate, min_weights, max_weights)
    elif method == "Gradient Descent (Mean-Variance)":
        weights = gradient_descent_optimize(expected_returns, cov_matrix, num_stocks, risk_free_rate, min_weights, max_weights)
    elif method == "SciPy (Constrained)":
        weights = scipy_optimize(expected_returns, cov_matrix, num_stocks, risk_free_rate, min_weights, max_weights)
    else:
        raise ValueError("Unknown method")
    
    # Calculate metrics, incorporating sentiment
    metrics = calculate_metrics(weights, expected_returns, cov_matrix, sentiment_scores, inflation, tax_rate, risk_free_rate, use_sharpe, use_inflation, use_tax_rate, use_advanced_metrics)
    
    return weights, metrics

def calculate_metrics(weights, expected_returns, cov_matrix, sentiment_scores, inflation, tax_rate, risk_free_rate, use_sharpe, use_inflation, use_tax_rate, use_advanced_metrics):
    metrics = {}
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) + 1e-10)
    portfolio_sentiment = np.dot(weights, sentiment_scores)
    
    metrics['Portfolio Return'] = portfolio_return
    metrics['Portfolio Volatility'] = portfolio_volatility
    metrics['Portfolio Sentiment Score'] = portfolio_sentiment
    
    if use_inflation:
        metrics['Real Return'] = portfolio_return - inflation
    if use_tax_rate:
        after_tax_return = np.sum(weights * (expected_returns * (1 - tax_rate)))
        metrics['After-Tax Return'] = after_tax_return
    if use_sharpe:
        metrics['Sharpe Ratio'] = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0
    
    if use_advanced_metrics:
        metrics['VaR 5%'] = portfolio_return - 1.645 * portfolio_volatility
        downside_vol = portfolio_volatility / np.sqrt(2)
        metrics['Sortino Ratio'] = (portfolio_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0
    
    return metrics

def monte_carlo_optimize(returns, cov, iterations, num_stocks, rf_rate, min_weights, max_weights):
    best_sharpe = -np.inf
    best_weights = None
    for _ in range(iterations):
        weights = np.random.random(num_stocks)
        if min_weights is not None:
            weights = np.clip(weights, min_weights, max_weights if max_weights is not None else np.inf)
        weights /= weights.sum() + 1e-10
        port_ret = np.dot(weights, returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)) + 1e-10)
        sharpe = (port_ret - rf_rate) / port_vol if port_vol > 0 else -np.inf
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights
    return best_weights

def genetic_algorithm_optimize(returns, cov, iterations, num_stocks, rf_rate, min_weights, max_weights):
    population_size = 50
    population = [np.random.dirichlet(np.ones(num_stocks)) for _ in range(population_size)]
    
    def fitness(w):
        if min_weights is not None:
            w = np.clip(w, min_weights, max_weights if max_weights is not None else np.inf)
            w /= w.sum() + 1e-10
        port_ret = np.dot(w, returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)) + 1e-10)
        return (port_ret - rf_rate) / port_vol if port_vol > 0 else -np.inf
    
    for _ in range(iterations // population_size):
        fitness_scores = [fitness(w) for w in population]
        parents_indices = np.argsort(fitness_scores)[-population_size//2:]
        parents = [population[i] for i in parents_indices]
        
        new_population = []
        for _ in range(population_size):
            p1, p2 = np.random.choice(parents, 2)
            child = (p1 + p2) / 2
            child /= child.sum() + 1e-10
            if np.random.random() < 0.1:
                child += np.random.normal(0, 0.02, num_stocks)
                child = np.clip(child, 0, 1)
                child /= child.sum() + 1e-10
            new_population.append(child)
        population = new_population
    
    best_idx = np.argmax([fitness(w) for w in population])
    return population[best_idx]

def gradient_descent_optimize(returns, cov, num_stocks, rf_rate, min_weights, max_weights, max_iter=1000, lr=0.01):
    weights = np.ones(num_stocks) / num_stocks
    
    for iteration in range(max_iter):
        port_ret = np.dot(weights, returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)) + 1e-10)
        
        if port_vol < 1e-8:
            break
            
        sharpe = (port_ret - rf_rate) / port_vol
        grad_ret = returns
        grad_vol = np.dot(cov, weights) / port_vol
        grad_sharpe = (grad_ret * port_vol - (port_ret - rf_rate) * grad_vol) / (port_vol ** 2)
        
        weights = weights + lr * grad_sharpe
        if min_weights is not None:
            weights = np.clip(weights, min_weights, max_weights if max_weights is not None else np.inf)
        weights /= weights.sum() + 1e-10
    
    return weights

def scipy_optimize(returns, cov, num_stocks, rf_rate, min_weights, max_weights):
    def objective(w):
        port_ret = np.dot(w, returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)) + 1e-10)
        return - (port_ret - rf_rate) / port_vol
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = list(zip(min_weights or [0]*num_stocks, max_weights or [1]*num_stocks))
    initial = np.ones(num_stocks) / num_stocks
    result = opt.minimize(objective, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    return result.x

def genetic_optimize_sentiment_weights(features, scores, generations=50):
    def fitness_func(ga_instance, solution, solution_idx):
        calc_scores = np.dot(features, solution)
        bull = calc_scores[scores > 0]
        bull_mean = np.mean(bull) if len(bull) > 0 else 0
        bear = calc_scores[scores < 0]
        bear_mean = np.mean(bear) if len(bear) > 0 else 0
        return abs(bull_mean - bear_mean)
    
    ga_instance = pygad.GA(num_generations=generations,
                           num_parents_mating=4,
                           fitness_func=fitness_func,
                           sol_per_pop=20,
                           num_genes=5,
                           init_range_low=-1.0,
                           init_range_high=1.0)
    ga_instance.run()
    return ga_instance.best_solution()[0]
