 
import numpy as np
import pandas as pd

def transform_feature(
    df: pd.DataFrame,
    col: str,
    target= str
) -> pd.DataFrame:
    """
    Apply transformations (log, sqrt) to a single feature
    and return a comparison DataFrame with 'transformation' as the first column.
    """
    # Log transform
    log_trans = np.log1p(df[col])

    # Square root transform
    sqrt_trans = np.sqrt(df[col])

    # Collect results with clean transformation names
    features = {
        "original": df[col],
        "log": log_trans,
        "sqrt": sqrt_trans
    }

    results = []
    for name, series in features.items():
        # Outliers
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        outliers_pct = round(outliers / len(series), 4)

        results.append({
            "transformation": name,
            "mean": round(series.mean(), 4),
            "std": round(series.std(), 4),
            "outliers_%": outliers_pct,
            "skew": round(series.skew(), 4),
            "corr_with_target": round(series.corr(df[target]), 4)
        })

    return pd.DataFrame(results)


def compare_transformations(
    df: pd.DataFrame,
    cols: list[str],
    target: str
) -> None:
    """
    Generate separate comparison tables for each feature in cols.
    Prints each table with headers.
    """
    cols = [c for c in cols if c != target]  # exclude target from output

    for col in cols:
        table = transform_feature(df, col, target)
        print(f"\n========== VARIABLE: {col} ==========\n")
        print(table)