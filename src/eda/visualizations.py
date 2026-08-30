import math
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
import seaborn as sns
import numpy as np


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
        #ax.tick_params(axis="x", rotation=0)
        labels = [textwrap.fill(label.get_text(), width=12) for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=0, fontsize=10)

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

    # Safely flatten axes regardless of grid dimensions ---
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        import numpy as np

        axes = np.atleast_1d(axes)

    # Calculate layout dimensions (always 2 columns)
    n_cols = 2
    n_rows = math.ceil(len(valid_features) / n_cols)

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
        #ax.tick_params(axis="x", rotation=0)
        labels = [textwrap.fill(label.get_text(), width=12) for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=0, fontsize=10)

        # Add percentage text annotations above bars
        for i, label in enumerate(churn_rates.index):
            rate = churn_rates[label]

            ax.text(
                i,
                rate + (churn_rates.max() * 0.01),
                f"{rate:.2f}%",
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

# ============================================================
# Continuous Variable Distributions (Histograms)
# ============================================================

def plot_numerical_distributions(df: pd.DataFrame, features: list, bins: int):
    """Plot frequency distributions and density curves for specified numerical features in a 2-column grid.

    Generates histograms annotated with mean, median, and descriptive
    statistics to visualize the spread and central tendency of continuous data.

    Parameters:
    -----------
    df : pd.DataFrame
        The source DataFrame.
    features : list
        List of numerical column names to plot.
    bins : int
        Number of bins used in the histograms.
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

    # Safely flatten axes regardless of grid dimensions
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = np.atleast_1d(axes)

    # Iterate through features and assign each to a specific subplot axis
    for idx, feature in enumerate(valid_features):
        ax = axes[idx]

        # Drop NaN values for accurate distribution plotting
        data = df[feature].dropna()

        # Plot histogram (absolute frequency)
        ax.hist(data, bins=bins, color="#1f77b4", edgecolor="white", alpha=0.8)

        # Calculate central tendency metrics
        mean_val = data.mean()
        median_val = data.median()

        # Calculate descriptive statistics
        min_val = data.min()
        p25 = data.quantile(0.25)
        p50 = data.quantile(0.50)
        p75 = data.quantile(0.75)
        p90 = data.quantile(0.90)
        max_val = data.max()

        # Add vertical reference lines for Mean and Median
        ax.axvline(
            mean_val,
            color="red",
            linestyle="dashed",
            linewidth=1.5,
            label=f"Mean: {mean_val:,.1f}",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle="dotted",
            linewidth=2,
            label=f"Median: {median_val:,.1f}",
        )

        # Add distribution statistics box
        stats_text = (
            f"Min: {min_val:,.2f}\n"
            f"P25: {p25:,.2f}\n"
            f"P50: {p50:,.2f}\n"
            f"P75: {p75:,.2f}\n"
            f"P90: {p90:,.2f}\n"
            f"Max: {max_val:,.2f}"
        )

        ax.text(
            0.98,
            0.84,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.2),
        )

        # Formatting titles, labels and legend
        ax.set_title(
            f"{feature.capitalize()} Numerical Distribution",
            fontsize=12,
            pad=15,
        )
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Number of Customers", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)

        # Ensure grid lines are visible on the Y-axis for better readability
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Clean up and hide any unused remaining axes in the grid
    for idx in range(len(valid_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

# ============================================================
# Continuous vs. Categorical Distributions (Boxplots)
# ============================================================

def plot_continuous_by_categorical(df: pd.DataFrame, continuous_features: list, categorical_feature: str):
    """Plot boxplots with overlaid swarm/strip indications for continuous features grouped by a categorical variable.

    Generates stratified distributions for each continuous column in the features list
    against a common target categorical baseline in a 2-column grid.

    Parameters:
    -----------
    df : pd.DataFrame
        The source DataFrame.
    continuous_features : list
        List of continuous column names to plot on the Y-axis.
    categorical_feature : str
        The categorical column name used for grouping on the X-axis.
    """
    # Filter out features that are not in the DataFrame
    if categorical_feature not in df.columns:
        print(f"Warning: Categorical feature '{categorical_feature}' not found in DataFrame.")
        return

    valid_continuous = [f for f in continuous_features if f in df.columns]

    if not valid_continuous:
        print("Warning: No valid continuous features found to plot.")
        return

    # Calculate layout dimensions (always 2 columns)
    n_cols = 2
    n_rows = math.ceil(len(valid_continuous) / n_cols)

    # Initialize grid figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))

    # Safely flatten axes regardless of grid dimensions ---
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        import numpy as np

        axes = np.atleast_1d(axes)


    # Iterate through continuous features and assign each to a specific subplot axis
    for idx, cont_feature in enumerate(valid_continuous):
        ax = axes[idx]

        # Plot distribution using seaborn boxplot
        sns.boxplot(
            data=df,
            x=categorical_feature,
            y=cont_feature,
            ax=ax,
            hue=categorical_feature,  
            legend=False,             
            palette=["#1f77b4", "#ff7f0e"],
            fliersize=4,
            width=0.5
        )

        # Formatting titles and labels
        ax.set_title(
            f"{cont_feature.capitalize()} Distribution by {categorical_feature.capitalize()}",
            fontsize=12,
            pad=15,
        )
        ax.set_xlabel(categorical_feature.capitalize(), fontsize=10)
        ax.set_ylabel(cont_feature.capitalize(), fontsize=10)
        
        # Format X-axis ticks to handle long category texts neatly
        ticks = ax.get_xticks()
        ax.set_xticks(ticks) 
        
        labels = [textwrap.fill(label.get_text(), width=12) for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=0, fontsize=10)

    # Clean up and hide any unused remaining axes in the grid
    for idx in range(len(valid_continuous), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


# ============================================================
# Churn Rate by Feature Interaction
# ============================================================

def plot_churn_rate_by_group(
    df: pd.DataFrame,
    x_feature: str,
    hue_feature: str,
    target: str = "Churn_bool",
):
    """Plot churn rate across groups of a feature, separated by another categorical feature.

    Calculates the churn rate for each combination of the specified
    grouping variables and displays the results as a line plot.

    Parameters:
    -----------
    df : pd.DataFrame
        The source DataFrame.
    x_feature : str
        Categorical feature displayed on the x-axis.
    hue_feature : str
        Categorical feature used to create separate lines.
    target : str, default="Churn_bool"
        Binary target variable used to calculate churn rate.
    """

    # Validate that all required columns exist in the DataFrame
    required_features = [x_feature, hue_feature, target]
    missing_features = [f for f in required_features if f not in df.columns]

    if missing_features:
        print(
            f"Warning: The following columns were not found in the DataFrame: "
            f"{missing_features}"
        )
        return

    # Calculate churn rate for each combination of the grouping variables
    churn_rate = (
        df.groupby([x_feature, hue_feature], observed=False)[target]
        .mean()
        .mul(100)
        .reset_index()
    )

    # Create the line plot
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=churn_rate,
        x=x_feature,
        y=target,
        hue=hue_feature,
        marker="o",
        linewidth=2,
    )

    # Add churn rate values to each data point
    for _, row in churn_rate.iterrows():
        plt.text(
            row[x_feature],
            row[target] + 1,
            f"{row[target]:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Format plot
    plt.title(
        f"Churn Rate by {x_feature} and {hue_feature}",
        fontsize=12,
        pad=15,
    )
    plt.xlabel(x_feature, fontsize=10)
    plt.ylabel("Churn Rate (%)", fontsize=10)

    plt.legend(
        title=hue_feature,
        fontsize=9,
        title_fontsize=9,
    )

    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()
