import math
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Frequency Distribution (Absolute & Relative)
# ============================================================

def plot_categorical_distributions(df: pd.DataFrame, features: list):
    """Plot absolute counts and percentages for specified categorical features in a 2-column grid.

    Calculates value frequencies and total population ratios for each column in
    the features list, displaying them as annotated bar charts.

    Parameters:
    -----------
    df : pd.DataFrame
        The source DataFrame.
    features : list
        List of categorical column names to plot.
    """
    # Filter out features that are not in the DataFrame
    valid_features = [f for f in features if f in df.columns]

    if not valid_features:
        print("Warning: No valid features found to plot.")
        return

    # Calculate layout dimensions (always 2 columns)
    n_cols = 2
    n_rows = math.ceil(len(valid_features) / n_cols)

    # Initialize grid figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))

    # Ensure axes is a flat array even if there is only 1 row or 1 plot
    if n_rows == 1:
        axes = axes.flatten() if len(valid_features) > 1 else [axes]
    else:
        axes = axes.flatten()

    # Iterate through features and assign each to a specific subplot axis
    for idx, feature in enumerate(valid_features):
        ax = axes[idx]

        # Calculate metrics
        counts = df[feature].value_counts()
        percentages = df[feature].value_counts(normalize=True).mul(100)

        # Plot absolute frequencies
        counts.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"])

        # Formatting titles and labels
        ax.set_title(
            f"{feature.capitalize()} Distribution (Frequency and %)",
            fontsize=12,
            pad=15,
        )
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Number of Customers", fontsize=10)
        ax.tick_params(axis="x", rotation=0)

        # Add metric text annotations to bars
        for i, label in enumerate(counts.index):
            count = counts[label]
            pct = percentages[label]

            ax.text(
                i,
                count + (counts.max() * 0.01),
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_ylim(0, counts.max() * 1.15)

    # Clean up and hide any unused remaining axes in the grid
    for idx in range(len(valid_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


# ============================================================
# Churn Rate by Category
# ============================================================

def plot_categorical_churn_rates(df: pd.DataFrame, features: list, target: str):
    """Plot churn rates for specified categorical features in a 2-column grid.

    Calculates the percentage of churn within each category of the features
    list and displays them as annotated bar charts.

    Parameters:
    -----------
    df : pd.DataFrame
        The source DataFrame.
    features : list
        List of categorical column names to plot.
    """
    # Filter out features that are not in the DataFrame
    valid_features = [f for f in features if f in df.columns]

    if not valid_features or target not in df.columns:
        print("Warning: No valid features or target found to plot.")
        return

    # Calculate layout dimensions (always 2 columns)
    n_cols = 2
    n_rows = math.ceil(len(valid_features) / n_cols)

    # Initialize grid figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))

    # Ensure axes is a flat array even if there is only 1 row or 1 plot
    if n_rows == 1:
        axes = axes.flatten() if len(valid_features) > 1 else [axes]
    else:
        axes = axes.flatten()

    # Iterate through features and assign each to a specific subplot axis
    for idx, feature in enumerate(valid_features):
        ax = axes[idx]

        # Calculate the mean target rate per category and convert to percentage
        churn_rates = df.groupby(feature)[target].mean().mul(100)

        # Plot churn rates using forest green and matching terracotta
        churn_rates.plot(kind="bar", ax=ax, color = ["#16B170", "#861f99"])

        # Formatting titles and labels
        ax.set_title(
            f"{feature.capitalize()} Churn Rate (%)", fontsize=12, pad=15
        )
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Churn Rate (%)", fontsize=10)
        ax.tick_params(axis="x", rotation=0)

        # Add percentage text annotations above bars
        for i, label in enumerate(churn_rates.index):
            rate = churn_rates[label]

            ax.text(
                i,
                rate + (churn_rates.max() * 0.01),
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Adjust y-axis limit to prevent label clipping at the top
        ax.set_ylim(0, churn_rates.max() * 1.15)

    # Clean up and hide any unused remaining axes in the grid
    for idx in range(len(valid_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()
