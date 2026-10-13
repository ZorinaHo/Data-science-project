"""Reproducible housing-price regression demo with synthetic data.

The data are illustrative, not actual housing transactions. Run:
    python analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / 'visualizations'
DATA = ROOT / 'data'
FIGURES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
RANDOM_STATE = 42


def make_demo_data(n=600, seed=RANDOM_STATE):
    """Create artificial house listings; no personal records or external downloads."""
    rng = np.random.default_rng(seed)
    floor_area = rng.uniform(55, 320, n)
    bedrooms = np.clip(np.rint(floor_area / 48 + rng.normal(0, 0.9, n)), 1, 7)
    age = rng.uniform(0, 65, n)
    distance = rng.uniform(1, 36, n)
    # Deliberate nonlinear premium at the high end makes residual diagnostics meaningful.
    luxury_premium = 0.012 * np.maximum(floor_area - 225, 0) ** 2
    price = (80 + 2.35 * floor_area + 22 * bedrooms - 1.9 * age
             - 8.0 * distance + luxury_premium
             + rng.normal(0, 55 + 0.15 * floor_area, n))
    return pd.DataFrame({
        'floor_area_sqm': np.round(floor_area, 1),
        'bedrooms': bedrooms.astype(int),
        'building_age_years': np.round(age, 1),
        'distance_to_center_km': np.round(distance, 1),
        'price_usd_thousands': np.maximum(price, 25).round(1),
    })


def main():
    df = make_demo_data()
    df.to_csv(DATA / 'synthetic_housing.csv', index=False)
    cols = ['floor_area_sqm', 'bedrooms', 'building_age_years', 'distance_to_center_km']
    X = df[cols]
    y = df['price_usd_thousands']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    models = {
        'Linear regression': make_pipeline(StandardScaler(), LinearRegression()),
        'Polynomial ridge (degree 3)': make_pipeline(StandardScaler(), PolynomialFeatures(degree=3, include_bias=False), Ridge(alpha=10)),
    }
    results = []
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        preds[name] = test_pred
        results.append({
            'model': name,
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'test_mae_usd_thousands': mean_absolute_error(y_test, test_pred),
            'test_rmse_usd_thousands': np.sqrt(mean_squared_error(y_test, test_pred)),
        })
    report = pd.DataFrame(results)
    report.to_csv(ROOT / 'model_results.csv', index=False)

    # Figure 1: actual vs predicted for both models on the held-out test split.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharex=True, sharey=True)
    lo = min(y_test.min(), *(p.min() for p in preds.values()))
    hi = max(y_test.max(), *(p.max() for p in preds.values()))
    for ax, (name, pred) in zip(axes, preds.items()):
        ax.scatter(y_test, pred, alpha=0.55, s=25, color='#3975AA')
        ax.plot([lo, hi], [lo, hi], '--', color='#C05746', linewidth=1.5)
        ax.set(title=name, xlabel='Actual price (USD thousands)', ylabel='Predicted price (USD thousands)')
        ax.grid(alpha=0.17)
    fig.suptitle('Actual vs Predicted: Held-out Test Data')
    fig.tight_layout()
    fig.savefig(FIGURES / 'actual_vs_predicted.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    # Figure 2: residual pattern across prediction range, including high-value homes.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, pred in preds.items():
        ax.scatter(pred, y_test.to_numpy() - pred, alpha=0.5, s=26, label=name)
    ax.axhline(0, color='#C05746', linestyle='--', linewidth=1.6)
    ax.set(title='Residual Diagnostics on Held-out Test Data', xlabel='Predicted price (USD thousands)',
           ylabel='Residual = actual - predicted (USD thousands)')
    ax.legend()
    ax.grid(alpha=0.17)
    fig.tight_layout()
    fig.savefig(FIGURES / 'residual_diagnostics.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(report.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\nSaved two figures, synthetic data, and model_results.csv.')


if __name__ == '__main__':
    main()
