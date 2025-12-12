#!/usr/bin/env python3
# coding: utf-8

"""
Supports analysis for heart disease classification.

Import and clean UCI Heart data.

GallantlyStellar
"""

import matplotlib.pyplot as plt
import pandas as pd


def importUCI(path: str = "../assets/data/raw") -> pd.DataFrame:
    """Import the 4 data files and return a dataframe"""
    # read in data, append origin marker to index so there are no duplicated indices
    dfva = pd.read_csv(path + "/processed.va.data", na_values="?", header=None)
    dfva.index = dfva.index.astype(str) + "va"
    dfhu = pd.read_csv(path + "/processed.hungarian.data", na_values="?", header=None)
    dfhu.index = dfhu.index.astype(str) + "hu"
    dfcl = pd.read_csv(path + "/processed.cleveland.data", na_values="?", header=None)
    dfcl.index = dfcl.index.astype(str) + "cl"
    dfsw = pd.read_csv(path + "/processed.switzerland.data", na_values="?", header=None)
    dfsw.index = dfsw.index.astype(str) + "sw"

    df = pd.concat([dfva, dfhu, dfcl, dfsw])
    del dfva, dfhu, dfcl, dfsw

    # Name columns
    df.columns = [
        "age",
        "isMale",
        "cp",
        "restSBP",
        "chol",
        "fastingBGLHigh",
        "restECG",
        "exerciseMaxHR",
        "exerciseAngina",
        "stDepressionExercise",
        "stSlope",
        "caFluor",
        "thal",
        "stage",
    ]
    # Set dtypes for memory efficiency (and nullabiliby needed)
    df = df.astype(
        {
            "age": "uint8",
            "isMale": "uint8",
            # "cp" currently label encoded
            "restSBP": "Int16",
            "chol": "Int16",
            "fastingBGLHigh": "Int8",
            # "restECG" currently label encoded
            "exerciseMaxHR": "Int16",
            "exerciseAngina": "Int8",
            "stDepressionExercise": "float32",
            # "stSlope" currently label encoded
            "caFluor": "Int8",
            # "thal" currently label encoded
            "stage": "uint8",
        }
    )
    # Denormalize label encodes for clarity and to facilitate one-hot encoding
    df["cp"] = (
        df["cp"]
        .replace(
            {
                1: "typical anginal",
                2: "atypical anginal",
                3: "nonanginal",
                4: "asymptomatic",
            }
        )
        .astype("category")
    )
    df["restECG"] = (
        df["restECG"].replace({0: "normal", 1: "ST-T abnormality", 2: "LVH"}).astype("category")
    )
    df["stSlope"] = (
        df["stSlope"].replace({1: "upsloping", 2: "flat", 3: "downsloping"}).astype("category")
    )
    df["thal"] = df["thal"].replace({3: "fixed", 6: "reversible", 7: "none"}).astype("category")

    # 2 duplicated rows
    df.drop_duplicates(inplace=True)
    return df


def msnoMatrix(df: pd.DataFrame) -> plt.Axes:
    """Create a missing value matrix plot."""
    from missingno import matrix

    plt.rcParams.update({"font.size": 20})  # bigger plot fonts
    # num pts from each site
    lenva = df.index.str.contains("va").sum()  # VA in Long Beach
    lenhu = df.index.str.contains("hu").sum()  # Hungary
    lencl = df.index.str.contains("cl").sum()  # Cleveland
    lensw = df.index.str.contains("sw").sum()  # Switzerland

    # offsets of 0.5 are due to msno matrix defaults
    start = -0.5
    stop = lenva - 0.5

    # create msno matrix plot
    ax = matrix(df)
    # shade each region
    ax.axhspan(start, stop, color="#009E73", alpha=0.3, label="VA")
    start += stop - start
    stop += lenhu
    ax.axhspan(start, stop, color="#D55E00", alpha=0.3, label="Hungary")
    start += stop - start
    stop += lencl
    ax.axhspan(start, stop, color="#56B4E9", alpha=0.3, label="Cleveland")
    start += stop - start
    stop += lensw
    ax.axhspan(start, stop, color="#CC79A7", alpha=0.3, label="Switzerland")
    # position legend below, centered
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncols=4)
    ax.set_ylabel("Patient")
    return ax


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Impute problematic values in the UCI dataset."""
    # quantitative values as median (to resist outliers)
    for quantCol in ["restSBP", "chol", "exerciseMaxHR", "stDepressionExercise"]:
        df.loc[df[quantCol].isna(), quantCol] = round(df[quantCol].median())

    # qualitative (and CA fluoroscopy due to only 3 discrete values) as mode
    for qualCol in [
        "fastingBGLHigh",
        "restECG",
        "exerciseAngina",
        "stSlope",
        "caFluor",
        "thal",
    ]:
        df.loc[df[qualCol].isna(), qualCol] = df[qualCol].mode()[0]

    # inappropriate values
    df.loc[df["restSBP"] == 0, "restSBP"] = round(df["restSBP"].mean())
    df.loc[df["chol"] == 0, "chol"] = round(df["chol"].mean())
    return df


def resetDtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set np (not pd) dtypes post-imputation of N/A values.

    Statsmodels requires this.
    """
    df = df.astype(
        {
            "restSBP": "uint8",
            "chol": "uint16",
            "fastingBGLHigh": "uint8",
            "exerciseMaxHR": "uint8",
            "exerciseAngina": "uint8",
            "caFluor": "uint8",
        }
    )
    return df
