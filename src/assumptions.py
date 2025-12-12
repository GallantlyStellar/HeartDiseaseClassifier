#!/usr/bin/env python3
# coding: utf-8

"""
Supports analysis for heart disease classification.

Test the assumptions of logistic regression.

GallantlyStellar
"""

from typing import List

import pandas as pd
from statsmodels.base.wrapper import ResultsWrapper


def assumptionsCheck(df: pd.DataFrame, models: List[ResultsWrapper]) -> None:
    """Check 6 assumptions of logistic regression."""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from statsmodels.api import MNLogit

    # Assumption 1: binary response levels
    # asserted in logistic.py on line 60 (while training each model)
    #
    # Assumption 2: independence
    assert ~df.duplicated().any(), "Check for duplicates/independence of rows"

    # Independence of residuals:
    # Studentized residuals
    fig, axs = plt.subplots(nrows=2, ncols=5)
    fig.supxlabel("Row Number", y=0.05)  # bottom row only
    fig.text(
        x=0.5,
        y=0.53,
        s="Linear Predictor Value",
        ha="center",
        fontsize="large",
    )
    fig.supylabel("Studentized Pearson Residual", x=0.08)
    axs = axs.T.reshape(-1)  # flatten, transpose
    modelNum = 0
    plot = 0
    for model in models:
        # https://www.pythonfordatascience.org/logistic-regression-python/
        # look for horizontal line with zero intercept
        sns.regplot(
            x=model.fittedvalues,
            y=model.resid_pearson,
            ax=axs[plot],
            color="black",
            scatter_kws={"s": 5},  # make smaller
            line_kws={"color": "green"},
            lowess=True,
        )
        axs[plot].axhline(0, color="red", linestyle="--")

        # axs[plot].set_xlabel("Linear Predictor Value")
        # axs[plot].set_ylabel("Studentized Pearson Residual")
        axs[plot].spines[["right", "top"]].set_visible(False)
        plot += 1

        # Plot residuals as a series
        axs[plot].plot(range(len(model.resid_pearson)), model.resid_pearson)
        # axs[plot].set_xlabel("Row Number")
        # axs[plot].set_ylabel("Studentized Pearson Residual")
        # axs[plot].set_title(f"Residuals for all rows for model {modelNum}")
        axs[plot].spines[["right", "top"]].set_visible(False)
        axs[plot].axhline(0, color="red", linestyle="--")

        plot += 1
        modelNum += 1

    # Assumption 3: multicolinearity
    fig, ax = plt.subplots()
    ax = sns.heatmap(
        df.corr(),
        annot=True,
        fmt="0.1f",  # 1 decimal on annotations
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=5,  # space between items
        xticklabels=True,  # show every label
        yticklabels=True,
    )
    ax.collections[0].colorbar.set_label(
        "Correlation Coefficient",
        rotation=-90,
        va="bottom",
    )
    ax.xaxis.tick_top()  # Put x labels above
    ax.set_xlabel("Column abbreviation")
    ax.set_ylabel("Column abbreviation")
    ax.xaxis.set_label_position("top")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-30, ha="right")
    ax.spines[:].set_visible(False)

    # Assumption 4: extreme outliers
    fig, axs = plt.subplots(nrows=2, ncols=3)
    axs = axs.reshape(-1)  # flatten
    fig.delaxes(axs[5])
    i = 0
    for model in models:
        ax = axs[i]
        cooks = pd.Series(model.get_influence().cooks_distance[0]).sort_values(ascending=False)
        ax.hist(
            cooks,
            color=plt.get_cmap("viridis", 5)(i),
            bins=20,
        )
        ax.axvline(4 / model.nobs, color="red")
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_title(f"Model {i}")
        i += 1

        outlierProp = sum(cooks > 4 / model.nobs) / cooks.shape[0]
        print(f"Proportion of influential values for model {i}: {outlierProp}")
        del cooks, outlierProp

    # Assumption 5: linearity between exog and logit of endog
    # Using Box-Tidwell test
    df_augmented = df.copy(deep=True)
    for column in df_augmented.columns:
        if df_augmented[column].nunique() > 10:  # continuous values only
            df_augmented[column + "_log"] = np.log(df_augmented[column])
    df_augmented = df_augmented.replace([np.inf, -np.inf], np.nan).dropna()
    model_box_tidwell = MNLogit(
        df_augmented["stage"],
        df_augmented.drop(
            "stage",
            axis=1,
        ),
    ).fit()
    pval_bt = model_box_tidwell.pvalues[model_box_tidwell.pvalues.index.str.endswith("_log")]
    print("\np-values for Box-Tidwell test:")
    print(pval_bt)
    assert ~(pval_bt <= 0.05).any().any(), "Box-Tidwell test failed"
    del df_augmented, model_box_tidwell

    # Assumption 6: sufficient size for each level to predict
    assert (df["stage"].value_counts() >= 10).all(), "Insufficient number of rows"

    return None
