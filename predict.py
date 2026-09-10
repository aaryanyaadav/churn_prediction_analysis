import joblib
import pandas as pd


# ==================================================
# LOAD MODEL ARTIFACTS
# ==================================================

model = joblib.load(
    "models/churn_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

encoder = joblib.load(
    "models/encoder.pkl"
)


# ==================================================
# PREPROCESS DATA
# ==================================================

def preprocess_data(df):

    df = df.copy()


    # -----------------------------------------------
    # CATEGORICAL FEATURES
    # -----------------------------------------------

    categorical_columns = list(
        encoder.feature_names_in_
    )

    encoded_data = encoder.transform(
        df[categorical_columns]
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(
            categorical_columns
        ),
        index=df.index
    )


    # -----------------------------------------------
    # NUMERICAL FEATURES
    # -----------------------------------------------

    numerical_columns = list(
        scaler.feature_names_in_
    )

    scaled_data = scaler.transform(
        df[numerical_columns]
    )

    scaled_df = pd.DataFrame(
        scaled_data,
        columns=numerical_columns,
        index=df.index
    )


    # -----------------------------------------------
    # UN-SCALED FEATURES
    # -----------------------------------------------

    unscaled_columns = [
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "ActiveWithCard"
    ]

    unscaled_df = df[
        unscaled_columns
    ].copy()


    # -----------------------------------------------
    # COMBINE EVERYTHING
    # -----------------------------------------------

    final_data = pd.concat(
        [
            scaled_df,
            unscaled_df,
            encoded_df
        ],
        axis=1
    )


    # -----------------------------------------------
    # EXACT MODEL ORDER
    # -----------------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        final_data = final_data[
            list(model.feature_names_in_)
        ]


    return final_data