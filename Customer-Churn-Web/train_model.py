import joblib

# Save model
joblib.dump(clf, "model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save encoders
joblib.dump(labelencoder, "label_encoder.pkl")

# Save feature columns
joblib.dump(list(X.columns), "columns.pkl")

print("All files saved successfully!")