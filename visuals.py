"""
visuals.py
----------
All the charts shown on the dashboard. Every function returns a matplotlib
figure, and the dashboard displays it with st.pyplot().
"""

import matplotlib

matplotlib.use("Agg")   # draw charts without opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Colours (one consistent set for the whole project)
BLUE = "#257de0"      # main colour
ORANGE = "#f47230"    # second colour (used for "lowers the score" / error lines)
INK = "#191919"       # main text
MUTED = "#666666"     # axis text
GRID = "#e6e6e6"      # light grid lines
CATEGORY_COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]

# Diverging colours for the correlation heatmap: orange (negative) -> grey -> blue (positive)
CORR_CMAP = LinearSegmentedColormap.from_list("corr", ["#e76624", "#f1f1f1", "#1b64be"])


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(width=8, height=4.5):
    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    fig.patch.set_facecolor("white")
    return fig, ax


def score_histogram(df: pd.DataFrame, target: str):
    """How many students got each range of scores."""
    fig, ax = _figure()
    ax.hist(df[target], bins=25, color=BLUE, edgecolor="white", linewidth=1.5)
    mean = df[target].mean()
    ax.axvline(mean, color=INK, linestyle="--", linewidth=1.5)
    ax.annotate(f"Average {mean:.1f}", xy=(mean, ax.get_ylim()[1] * 0.95), xytext=(6, 0),
                textcoords="offset points", color=INK, fontsize=9)
    _style(ax, f"Distribution of {target}", target, "Number of students")
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame):
    """How strongly each pair of number columns moves together (-1 to +1)."""
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    size = max(5.0, 0.8 * len(corr.columns) + 2)
    fig, ax = plt.subplots(figsize=(size + 1.5, size), dpi=110)
    image = ax.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
    labels = [str(c) for c in corr.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9, color=MUTED)
    ax.set_yticklabels(labels, fontsize=9, color=MUTED)
    if len(labels) <= 15:
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = corr.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > 0.6 else INK)
    ax.set_title("Correlation between number columns", loc="left", fontsize=12,
                 color=INK, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.ax.tick_params(labelsize=8, colors=MUTED)
    bar.outline.set_visible(False)
    fig.tight_layout()
    return fig


def feature_vs_target(df: pd.DataFrame, feature: str, target: str):
    """Scatter plot of one input against the score, with a straight trend line."""
    fig, ax = _figure()
    x = df[feature].astype(float)
    y = df[target].astype(float)
    ax.scatter(x, y, s=18, color=BLUE, alpha=0.45, edgecolors="none")
    if x.nunique() > 1:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        ax.plot(line_x, slope * line_x + intercept, color=ORANGE, linewidth=2,
                label=f"Trend line (slope {slope:.2f})")
        ax.legend(frameon=False, fontsize=9, loc="best")
    _style(ax, f"{feature} vs {target}", feature, target)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    return fig


def box_by_category(df: pd.DataFrame, category: str, target: str):
    """Compare the scores of different groups (e.g. male vs female)."""
    groups = df.groupby(category)[target]
    order = groups.median().sort_values(ascending=False).index.tolist()[:12]
    values = [df.loc[df[category] == g, target].values for g in order]
    fig, ax = _figure()
    box = ax.boxplot(values, patch_artist=True, widths=0.55,
                     medianprops={"color": INK, "linewidth": 1.5},
                     whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                     flierprops={"marker": "o", "markersize": 3, "markerfacecolor": MUTED,
                                 "markeredgecolor": "none", "alpha": 0.5})
    for patch in box["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.75)
        patch.set_edgecolor("white")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([str(o) for o in order], rotation=20 if len(order) > 4 else 0,
                       ha="right" if len(order) > 4 else "center")
    _style(ax, f"{target} by {category}", category, target)
    fig.tight_layout()
    return fig


def missing_values_chart(df: pd.DataFrame):
    """Bar chart of empty cells per column (before cleaning). Returns None if there are none."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values()
    if missing.empty:
        return None
    fig, ax = _figure(8, max(2.5, 0.45 * len(missing) + 1.2))
    ax.barh([str(i) for i in missing.index], missing.values, color=BLUE, height=0.6)
    for i, value in enumerate(missing.values):
        ax.text(value, i, f"  {value}", va="center", fontsize=9, color=INK)
    _style(ax, "Empty cells in each column (before cleaning)", "Number of empty cells")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    return fig


def actual_vs_predicted(y_true, y_pred, target: str):
    """Points close to the dashed line = good predictions."""
    fig, ax = _figure(7, 5)
    ax.scatter(y_true, y_pred, s=18, color=BLUE, alpha=0.5, edgecolors="none", label="Test students")
    low = min(np.min(y_true), np.min(y_pred))
    high = max(np.max(y_true), np.max(y_pred))
    ax.plot([low, high], [low, high], color=ORANGE, linestyle="--", linewidth=2, label="Perfect prediction")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Actual vs predicted score", f"Actual {target}", f"Predicted {target}")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    return fig


def residual_plot(y_true, y_pred, target: str):
    """Errors (actual - predicted). A good model has errors spread evenly around 0."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = _figure(7, 5)
    ax.scatter(y_pred, residuals, s=18, color=BLUE, alpha=0.5, edgecolors="none")
    ax.axhline(0, color=ORANGE, linestyle="--", linewidth=2)
    _style(ax, "Prediction errors (residuals)", f"Predicted {target}", "Actual - Predicted")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    return fig


def coefficient_chart(coefficients: pd.DataFrame, max_items: int = 15):
    """How much each input changes the score (per 1 unit increase)."""
    data = coefficients.head(max_items).iloc[::-1]
    colors = [BLUE if c >= 0 else ORANGE for c in data["Coefficient"]]
    fig, ax = _figure(8, max(3, 0.45 * len(data) + 1.3))
    ax.barh(data["Input"].astype(str), data["Coefficient"], color=colors, height=0.6)
    ax.axvline(0, color=MUTED, linewidth=1)
    _style(ax, "Effect of each input on the score  (blue = raises, orange = lowers)",
           "Change in score for +1 unit of the input")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    return fig
