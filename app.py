from src.data_preprocessing import load_data, preprocess
from src.train_model import train_model, save_model
from src.evaluate_model import evaluate

# Load dataset
df = load_data("data/dataset.csv")

# Preprocess
X_train, X_test, y_train, y_test, scaler = preprocess(df, "purchased")

# Train model
model = train_model(X_train, y_train)

# Save model
save_model(model)

# Evaluate
evaluate(model, X_test, y_test)
