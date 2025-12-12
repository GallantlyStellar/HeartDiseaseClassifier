#!/usr/bin/env python3
# coding: utf-8

"""
Make predictions on heart disease.

GallantlyStellar
"""

import matplotlib.pyplot as plt
import pandas as pd

from assumptions import assumptionsCheck
from cleaning import importUCI, impute, msnoMatrix, resetDtypes
from logistic import predict, testStages, trainMNL, trainStage
from miscFigs import bivariate, presenceMatrix, univariate

# Make plot fonts bigger
plt.rcParams.update({"font.size": 20})

df = importUCI()
univariate(df, 4, 4)
bivariate(df, "stage", 4, 4)
msnoMatrix(df)
df = impute(df)
df = resetDtypes(df)  # statsmodels-friendly np dtypes

df = pd.get_dummies(df, dtype="uint8", drop_first=False)  # one-hot encoding
# Not dropping first to drop the normalcy categories so coefficients
# changes from baseline by default (drop to prevent multicolinearity)
df = df.drop(
    [
        "cp_asymptomatic",
        "restECG_normal",
        "stSlope_flat",
        "thal_none",
    ],
    axis=1,
)
# df.to_csv("../assets/data/processed/one-hotCleanedConcatedImputed.csv")

modelMN = trainMNL(df)
# modelMN.summary()
model0 = trainStage(df, stage=0)
model1 = trainStage(df, stage=1)
model2 = trainStage(df, stage=2)
model3 = trainStage(df, stage=3)
model4 = trainStage(df, stage=4)
models = [model0, model1, model2, model3, model4]
resultsTest = testStages(df, models)

resultsAll = predict(df, models)

assumptionsCheck(df, models)
presenceMatrix(df, models)

plt.show()
