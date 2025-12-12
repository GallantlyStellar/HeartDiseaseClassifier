#!/usr/bin/env python3
# coding: utf-8

"""
Supports analysis for heart disease classification.

Create additional visualizations.

GallantlyStellar
"""

from typing import List

import pandas as pd
from matplotlib.figure import Figure
from statsmodels.base.wrapper import ResultsWrapper


def univariate(df: pd.DataFrame, n_rows: int = 4, n_cols: int = 4) -> Figure:
    """
    Create univariate visualizations of a df.

    Args:
        df (DataFrame): A Pandas DataFrame to visualize.
        n_rows (int): The number of rows to display in the final figure.
        n_cols (int): The number of columns to display in the final figure.

    Returns:
        fig (matplotlib Figure): Use plt.show() to access figure.

    """
    import matplotlib.pyplot as plt
    import numpy as np

    assert n_rows * n_cols >= len(df.columns), "Not enough plots for all cols"

    PLOT_COUNT = len(df.columns)
    n_drop = (n_rows * n_cols) - PLOT_COUNT

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    axs = axs.reshape(-1)  # flatten to iterate
    # remove 2 unneeded plots
    _ = [axs[i].remove() for i in range(-n_drop, 0)]

    fig.supylabel("Occurences", x=-0.0005)
    fig.supxlabel("Feature Value", y=-0.003)
    cmap = plt.get_cmap("tab20", PLOT_COUNT)
    color = [cmap(i) for i in range(0, PLOT_COUNT)]
    np.random.seed(6396)
    np.random.shuffle(color)
    for i in range(0, PLOT_COUNT):
        ax = axs[i]
        ax.set_title(df.columns[i])
        ax.spines[["right", "top"]].set_visible(False)
        col = df.iloc[:, i]
        colUniqueVals = col.drop_duplicates().sort_values()
        if col.nunique() > 10:  # continuous values
            ax.hist(col, color=color[i])
        # binary values
        elif colUniqueVals.dropna().isin([0, 1]).all():
            col = col.astype(str).replace(
                {
                    "0": "False",
                    "1": "True",
                    "<NA>": "NA",
                    "nan": "NA",
                }
            )
            val_counts = col.value_counts().sort_index()
            ax.bar(val_counts.index, val_counts.values, color=color[i])
            del val_counts
        # discrete, non-binary values
        else:
            col = col.astype(str).replace(
                {
                    "<NA>": "NA",
                    "nan": "NA",
                }
            )
            val_counts = col.value_counts().sort_index()
            labels = val_counts.index.astype(str)
            ax.bar(labels, val_counts.values, color=color[i])
            ax.set_xticks(range(len(labels)))  # suppress warning
            ax.set_xticklabels([label[:5] + "." if len(label) > 5 else label for label in labels])
            del val_counts, labels
        del col, colUniqueVals
    return fig


def bivariate(df: pd.DataFrame, target: str, n_rows: int = 4, n_cols: int = 4) -> Figure:
    """
    Create univariate visualizations of a df.

    Args:
        df (DataFrame): A Pandas DataFrame to visualize.
        target (str): String matching the col name to use as the response var.
        n_rows (int): The number of rows to display in the final figure.
        n_cols (int): The number of columns to display in the final figure.

    Returns:
        fig (matplotlib Figure): Use plt.show() to access figure.

    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    assert n_rows * n_cols >= (len(df.columns) - 1), "Not enough plots for all cols"
    PLOT_COUNT = len(df.columns) - 1  # don't plot target vs itself
    n_drop = (n_rows * n_cols) - PLOT_COUNT

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    axs = axs.reshape(-1)  # flatten to iterate
    # remove 2 unneeded plots
    _ = [axs[i].remove() for i in range(-n_drop, 0)]

    fig.supylabel("Occurences", x=-0.0005)
    fig.supxlabel("Feature Value", y=-0.003)
    for i in range(0, PLOT_COUNT):
        ax = axs[i]
        ax.set_title(df.columns[i])
        ax.spines[["right", "top"]].set_visible(False)
        col = df.iloc[:, i]
        colUniqueVals = col.drop_duplicates().sort_values()
        if col.nunique() > 10:  # continuous values
            sns.histplot(
                x=col,
                hue=df[target],
                multiple="stack",
                element="step",
                palette="pastel",
                ax=ax,
                edgecolor=None,
            )
            # ax.hist(
            #     [col[df[target] == level] for level in df[target].unique()],
            #     fill=False,
            #     histtype="step",  # elim vertical lines between bars
            #     stacked=True,  # to compare distributions
            #     edgecolor=color,
            #     label=df[target].unique(),
            # )
        # binary values
        elif colUniqueVals.dropna().isin([0, 1]).all():
            col = col.astype(str).replace(
                {
                    "0": "False",
                    "1": "True",
                    "<NA>": "NA",
                    "nan": "NA",
                }
            )
            if "NA" in col.values:
                sns.countplot(
                    x=col,
                    hue=df[target],
                    ax=ax,
                    palette="pastel",
                    order=["False", "True", "NA"],
                )
            else:
                sns.countplot(
                    x=col,
                    hue=df[target],
                    ax=ax,
                    palette="pastel",
                    order=["False", "True"],
                )
        # discrete, non-binary values
        else:
            col = col.astype(str).replace(
                {
                    "<NA>": "NA",
                    "nan": "NA",
                }
            )
            sns.countplot(x=col, hue=df[target], ax=ax, palette="pastel")
            labels = col.value_counts().sort_index().index.astype(str)
            ax.set_xticks(range(len(labels)))  # suppress warning
            ax.set_xticklabels([label[:5] + "." if len(label) > 5 else label for label in labels])
            del labels
        # remove local legends and axis labels to have only one on figure
        ax.get_legend().remove()
        ax.set_xlabel("")
        ax.set_ylabel("")
        del col, colUniqueVals
    # get legend from last plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title=target, ncol=df[target].nunique(), loc="upper center")
    return fig


def presenceMatrix(df: pd.DataFrame, models: List[ResultsWrapper]) -> Figure:
    """Plot table of parameters in models."""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Get all possible model parameters, add "const"
    cols = list(df.columns)
    cols.insert(0, "const")

    # Instantiate df with 0s, row/column labels
    colNames = [f"Stage {i}" for i in range(len(models))]
    colNames[0] = "No disease"
    presenceMatrix = pd.DataFrame(2, index=cols, columns=colNames, dtype="uint8")
    presenceMatrixAnnot = pd.DataFrame(2, index=cols, columns=colNames, dtype="float64")
    del colNames

    # Set pm to 1 and pma to e^B if parameter matches feature
    for i in range(len(models)):
        for col in cols:
            if col in models[i].params.index:
                presenceMatrix.loc[col, presenceMatrix.columns[i]] = 1
                presenceMatrixAnnot.loc[col, presenceMatrixAnnot.columns[i]] = np.exp(
                    models[i].params[col]
                )
            else:
                presenceMatrix.loc[col, presenceMatrix.columns[i]] = 0
                presenceMatrixAnnot.loc[col, presenceMatrixAnnot.columns[i]] = 0
        # presenceMatrix.iloc[:, i] = pd.Series(
        #     [1 if col in models[i].params.index else 0 for col in cols]
        # )
    # Round to 3 decimals, blank 0s to annotate heatmap
    presenceMatrixAnnot = (
        presenceMatrixAnnot.map(lambda x: f"{x:.3f}")
        .astype(str)
        .replace(
            {
                "0.000": "",
            }
        )
    )

    fig, ax = plt.subplots()
    ax = sns.heatmap(
        presenceMatrix,
        cmap=sns.color_palette(["#ce3134", "#497f40"]),
        vmin=0,
        vmax=1,
        linewidths=5,  # space between items
        xticklabels=True,  # show every label
        yticklabels=True,
        # square=True,
        annot=presenceMatrixAnnot,
        fmt="",  # default fmt only works on numerics
    )
    ax.xaxis.tick_top()  # Put x labels above
    ax.xaxis.set_label_position("top")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="right")
    ax.set_xlabel("Parameters used to differentiate stage")
    ax.set_ylabel("Explanatory variable abbreviation")
    ax.collections[0].colorbar.remove()
    return fig
