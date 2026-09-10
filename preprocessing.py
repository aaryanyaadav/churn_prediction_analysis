import pandas as pd


def feature_engineering(df):

    df = df.copy()

    # Remove columns not used by model
    df = df.drop(
        columns=[
            "RowNumber",
            "CustomerId",
            "Surname"
        ],
        errors="ignore"
    )

    # -------------------------------
    # Feature 1
    # -------------------------------

    df["BalancePerProduct"] = (
        df["Balance"] /
        df["NumOfProducts"].replace(0, 1)
    )

    # -------------------------------
    # Feature 2
    # -------------------------------

    df["ActiveWithCard"] = (
        df["IsActiveMember"] *
        df["HasCrCard"]
    )

    return df