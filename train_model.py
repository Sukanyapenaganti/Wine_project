import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
X, y = load_wine(return_X_y=True)

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model correctly
with open("wine_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved successfully")
