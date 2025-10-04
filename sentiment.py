import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return {'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0, 'Rho': 0}
    
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
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta / 365,  # Daily
        'Vega': vega / 100,    # Per 1%
        'Rho': rho / 100       # Per 1%
    }

def compute_sentiment(greeks, current_rate, predicted_rate, inflation, weights=[1.0, 0.5, -0.5, 0.3, 0.2]):
    # weights: [Delta, Gamma, Theta, Vega, Rho]
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
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels, kmeans.cluster_centers_
