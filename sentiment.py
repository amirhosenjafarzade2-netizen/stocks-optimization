import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes Greeks for a stock's option.
    Parameters:
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Implied volatility
        option_type: 'call' or 'put'
    Returns:
        Dictionary with Delta, Gamma, Theta, Vega, Rho
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0, 'Rho': 0}
    
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            delta = norm.cdf(d1) - 1
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T) + 1e-10)  # Avoid division by zero
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta / 365,  # Daily
            'Vega': vega / 100,    # Per 1%
            'Rho': rho / 100       # Per 1%
        }
    except Exception as e:
        return {'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0, 'Rho': 0}

def compute_sentiment(greeks, current_rate, predicted_rate, inflation, weights=[1.0, 0.5, -1.0, 0.5, 0.3]):
    """
    Compute sentiment score and label.
    weights: [Delta, Gamma, Theta, Vega, Rho]
    """
    score = (weights[0] * greeks['Delta'] +
             weights[1] * greeks['Gamma'] +
             weights[2] * greeks['Theta'] +
             weights[3] * greeks['Vega'] +
             weights[4] * greeks['Rho'])
    
    # Macro adjustment
    rate_change = predicted_rate - current_rate
    macro_factor = 1 - 0.05 * rate_change - 0.03 * inflation
    adjusted_score = score * macro_factor
    
    sentiment = 'Bullish' if adjusted_score > 0 else 'Bearish'
    return adjusted_score, sentiment

def cluster_stocks(features, num_clusters=3):
    """
    Cluster stocks based on Greeks features.
    Returns: labels, centers
    """
    kmeans = KMeans(n_clusters=min(num_clusters, len(features)), random_state=42)
    labels = kmeans.fit_predict(features)
    return labels, kmeans.cluster_centers_

def interpret_clusters(centers):
    """
    Interpret cluster centers to assign meaningful labels.
    """
    labels = []
    for center in centers:
        delta, gamma, theta, vega, rho = center
        if delta > 0.3:
            labels.append("Bullish High Delta")
        elif delta < -0.3:
            labels.append("Bearish High Delta")
        elif vega > 0.5:
            labels.append("High Volatility Sensitivity")
        elif abs(theta) > 0.05:
            labels.append("High Time Decay Risk")
        else:
            labels.append("Neutral")
    return labels
