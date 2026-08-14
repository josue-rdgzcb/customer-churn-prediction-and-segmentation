from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

# ============================================================
# Chi-Square Test and Cramer's V
# ============================================================

def _cramers_v(contingency_table: pd.DataFrame, chi2: float) -> float:
    """Calculate Cramer's V from a contingency table and Chi-Square statistic."""
    n = contingency_table.to_numpy().sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * min(r - 1, k - 1)))


def _association_strength(v: float) -> str:
    """Classify the strength of association based on Cramer's V."""
    if v < 0.10:
        return "Muy baja"
    elif v < 0.30:
        return "Baja"
    elif v < 0.50:
        return "Moderada"
    else:
        return "Fuerte"


def chi_square_analysis(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    """
    Perform Chi-Square tests of independence and calculate Cramer's V
    for multiple categorical features against a target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the features and target variable.
    features : list
        List of categorical feature names to analyze.
    target : str
        Name of the categorical target variable.

    Returns
    -------
    pd.DataFrame
        Summary table containing Chi-Square statistic, p-value,
        Cramer's V, and association strength for each feature.
    """
    results = []

    for feature in features:
        # Create the contingency table between the feature and target
        contingency_table = pd.crosstab(df[feature], df[target])

        # Perform Chi-Square test of independence
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        # Calculate Cramer's V
        v = _cramers_v(contingency_table, chi2)

        results.append({
            "Variable": feature,
            "Chi-Square": chi2,
            "p-value": p_value,
            "Cramer's V": v,
            "Asociación": _association_strength(v)
        })

    return pd.DataFrame(results)