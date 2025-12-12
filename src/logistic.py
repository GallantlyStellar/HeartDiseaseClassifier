#!/usr/bin/env python3
# coding: utf-8

"""
Supports analysis for heart disease classification.

Create, fit, and score models.

GallantlyStellar
"""

from typing import List

import pandas as pd

# type(model).__bases__[0].__bases__[0].__bases__
from statsmodels.base.wrapper import ResultsWrapper


def trainMNL(df: pd.DataFrame) -> ResultsWrapper:
    """Train a multinomial logistic regression model with statsmodels."""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from statsmodels.api import MNLogit, add_constant

    dfMN = df.copy(deep=True)  # don't mutate df
    predictors = dfMN.columns.drop("stage")  # columns for X
    X_train, X_test, y_train, y_test = train_test_split(
        dfMN[predictors],
        dfMN["stage"],
        test_size=0.2,
        random_state=1571,
        stratify=dfMN["stage"],
    )

    X_train = add_constant(X_train)
    model = MNLogit(y_train, X_train).fit()

    pred = model.predict(add_constant(X_test)).idxmax(axis=1)
    print("Test set accuracy:", accuracy_score(y_test, pred))
    return model


def trainStage(
    df: pd.DataFrame, stage: int, colDrop: List[str] = [], alpha: float = 0.05
) -> ResultsWrapper:
    """Train a single logistic regression model for a given stage."""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from statsmodels.api import Logit, add_constant

    # Predict if it belongs to a stage (0) or not (1)
    dfStage = df.copy(deep=True)  # don't mutate df
    dfStage["stageSource"] = dfStage["stage"]
    dfStage.loc[dfStage["stageSource"] == stage, "stage"] = 1
    dfStage.loc[dfStage["stageSource"] != stage, "stage"] = 0
    dfStage.drop(["stageSource"], axis=1, inplace=True)

    # Assumption 1: binary response levels
    assert dfStage["stage"].nunique() == 2, "Response levels must be binary"

    predictors = dfStage.columns.drop("stage")  # columns for X
    X_train, X_test, y_train, y_test = train_test_split(
        dfStage[predictors],
        dfStage["stage"],
        test_size=0.2,
        random_state=1571,
        stratify=dfStage["stage"],
    )

    # Droping cols here lets arbitrary cols, incl const
    # be dropped if needed
    X_train = add_constant(X_train).drop(colDrop, axis=1)
    X_test = add_constant(X_test).drop(colDrop, axis=1)
    model = Logit(y_train, X_train).fit()

    # Drop for p-values below alpha
    while model.pvalues.sort_values(ascending=False).iloc[0] >= alpha:
        X_train = X_train.drop(model.pvalues.sort_values(ascending=False).index[0], axis=1)
        X_test = X_test.drop(model.pvalues.sort_values(ascending=False).index[0], axis=1)
        del model
        model = Logit(y_train, X_train).fit()

    pred = model.predict(X_test).round()
    print(f"Test set accuracy for stage {stage}: {accuracy_score(y_test, pred)}")
    return model


def testStages(df: pd.DataFrame, models: List[ResultsWrapper]) -> pd.DataFrame:
    """Apply the models to a test set and use softmax to return probabilities."""
    from scipy.special import softmax
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from statsmodels.api import add_constant

    predictorsAll = df.columns.drop("stage")  # columns for X
    X_train, X_test, y_train, y_test = train_test_split(
        df[predictorsAll],
        df["stage"],
        test_size=0.2,
        random_state=1571,
        stratify=df["stage"],
    )

    # Training set
    stage = 0
    results = pd.DataFrame()
    for model in models:
        # This allows testing if model had constant dropped during training
        X_train_forThisModel = X_train.copy(deep=True)
        if "const" in model.params.index:
            X_train_forThisModel = add_constant(X_train_forThisModel)
        predictorColDrops = [
            item for item in X_train_forThisModel.columns if item not in model.params.index
        ]
        X_train_forThisModel = X_train_forThisModel.drop(predictorColDrops, axis=1)
        series = pd.Series(model.predict(X_train_forThisModel), name=stage)
        results = pd.concat([results, series], axis=1)
        stage += 1
        del series, X_train_forThisModel, predictorColDrops
    del stage
    results = results.apply(softmax, axis=1).apply(pd.Series)
    print(f"Combined model train accuracy: {accuracy_score(y_train, results.idxmax(axis=1))}")

    # Test set
    stage = 0
    results = pd.DataFrame()
    for model in models:
        # This allows testing if model had constant dropped during training
        X_test_forThisModel = X_test.copy(deep=True)
        if "const" in model.params.index:
            X_test_forThisModel = add_constant(X_test_forThisModel)
        predictorColDrops = [
            item for item in X_test_forThisModel.columns if item not in model.params.index
        ]
        X_test_forThisModel = X_test_forThisModel.drop(predictorColDrops, axis=1)
        series = pd.Series(model.predict(X_test_forThisModel), name=stage)
        results = pd.concat([results, series], axis=1)
        stage += 1
        del series, X_test_forThisModel, predictorColDrops
    del stage
    results = results.apply(softmax, axis=1).apply(pd.Series)
    print(f"Combined model test accuracy: {accuracy_score(y_test, results.idxmax(axis=1))}")
    return results


def predict(df: pd.DataFrame, models: List[ResultsWrapper]) -> pd.DataFrame:
    """Apply the models to a dataframe and use softmax to return probabilities.
    Like testStages without the train-test split."""
    from scipy.special import softmax
    from statsmodels.api import add_constant

    dfLocal = df.drop("stage", axis=1)  # columns for X
    stage = 0
    results = pd.DataFrame()
    for model in models:
        # This allows testing if model had constant dropped during training
        if "const" in model.params.index:
            dfLocal = add_constant(dfLocal)
        predictorColDrops = [item for item in dfLocal.columns if item not in model.params.index]
        dfLocal_forThisModel = dfLocal.drop(predictorColDrops, axis=1)
        series = pd.Series(model.predict(dfLocal_forThisModel), name=stage)
        results = pd.concat([results, series], axis=1)
        stage += 1
        del series, dfLocal_forThisModel, predictorColDrops
    del stage
    results = results.apply(softmax, axis=1).apply(pd.Series)
    return results
