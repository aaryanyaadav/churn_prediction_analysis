import joblib

# Load the saved scaler
scaler = joblib.load("models/scaler.pkl")

# Load the encoder
encoder = joblib.load("models/encoder.pkl")

# Load the model
model = joblib.load("models/churn_model.pkl")


print("SCALER FEATURES:")
print(scaler.feature_names_in_)

print("\nENCODER FEATURES:")
print(encoder.feature_names_in_)

print("\nMODEL FEATURES:")
if hasattr(model, "feature_names_in_"):
    print(model.feature_names_in_)
else:
    print("Model does not have feature_names_in_")