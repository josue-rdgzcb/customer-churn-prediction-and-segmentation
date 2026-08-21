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


# ============================================================
# Mann-Whitney U Test and Effect size
# ============================================================

from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np


def rank_biserial_correlation(u_stat, n1, n2):
    """
    Calculate the rank-biserial correlation effect size.

    Parameters
    ----------
    u_stat : float
        Mann-Whitney U statistic for the first group.
    n1 : int
        Number of observations in the first group.
    n2 : int
        Number of observations in the second group.

    Returns
    -------
    float
        Rank-biserial correlation coefficient.
    """
    return (2 * u_stat) / (n1 * n2) - 1


def effect_size_strength(r):
    """
    Classify the effect size according to Cohen's thresholds.

    Parameters
    ----------
    r : float
        Rank-biserial correlation coefficient.

    Returns
    -------
    str
        Effect size interpretation.
    """
    r = abs(r)

    if r < 0.10:
        return "Muy pequeño"
    elif r < 0.30:
        return "Pequeño"
    elif r < 0.50:
        return "Moderado"
    else:
        return "Grande"


def mann_whitney_analysis(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    """
    Perform Mann-Whitney U tests for numerical features against a binary target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the features and target variable.
    features : list
        List of numerical feature names to analyze.
    target : str
        Name of the binary target variable.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the U statistic, p-value, rank-biserial correlation,
        and effect size interpretation for each feature.
    """
    results = []

    for feature in features:
        # Separate observations according to the target groups
        group_no = df.loc[df[target] == 0, feature].dropna()
        group_yes = df.loc[df[target] == 1, feature].dropna()

        # Perform two-sided Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(
            group_no,
            group_yes,
            alternative="two-sided"
        )

        # Calculate rank-biserial correlation
        n1 = len(group_no)
        n2 = len(group_yes)
        r_rb = rank_biserial_correlation(u_stat, n1, n2)

        results.append({
            "Variable": feature,
            "U Statistic": u_stat,
            "p-value": p_value,
            "Rank-biserial r": r_rb,
            "Effect Size": effect_size_strength(r_rb)
        })

    return pd.DataFrame(results)