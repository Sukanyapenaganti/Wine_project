from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
X, y = load_wine(return_X_y=True)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model properly
with open("wine_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ wine_model.pkl saved successfully")