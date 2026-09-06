import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv("Iris.csv")

print(df.head())
print("Dataset shape:", df.shape)


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

X = df[features].values

class_mapping = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

y = df["Species"].map(class_mapping).values


# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First: 80% train, 20% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Then: split temporary 20% into
# 10% validation + 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))


# ============================================================
# 5. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

# IMPORTANT:
# Fit ONLY on training data
X_train = scaler.fit_transform(X_train)

# Validation and test use the SAME scaler
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ============================================================
# 6. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

y_val = torch.tensor(
    y_val,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


# ============================================================
# 7. DATASET + DATALOADER
# ============================================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)


# ============================================================
# 8. ANN MODEL
# ============================================================

class IrisANN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(4, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 3)
        )

    def forward(self, x):

        return self.network(x)


model = IrisANN()

print(model)


# ============================================================
# 9. LOSS FUNCTION
# ============================================================

loss_function = nn.CrossEntropyLoss()


# ============================================================
# 10. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# ============================================================
# 11. TRAINING
# ============================================================

epochs = 100

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        # -----------------------------
        # Forward pass
        # -----------------------------

        outputs = model(X_batch)

        # -----------------------------
        # Calculate loss
        # -----------------------------

        loss = loss_function(
            outputs,
            y_batch
        )

        # -----------------------------
        # Remove old gradients
        # -----------------------------

        optimizer.zero_grad()

        # -----------------------------
        # Backpropagation
        # -----------------------------

        loss.backward()

        # -----------------------------
        # Update weights
        # -----------------------------

        optimizer.step()

        total_loss += loss.item()


    # Average training loss
    avg_loss = total_loss / len(train_loader)


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    with torch.no_grad():

        val_outputs = model(X_val)

        val_predictions = torch.argmax(
            val_outputs,
            dim=1
        )

        val_accuracy = accuracy_score(
            y_val.numpy(),
            val_predictions.numpy()
        )


    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Validation Accuracy: {val_accuracy:.2%}"
        )


# ============================================================
# 12. FINAL TEST
# ============================================================

model.eval()

with torch.no_grad():

    test_outputs = model(X_test)

    test_predictions = torch.argmax(
        test_outputs,
        dim=1
    )


test_accuracy = accuracy_score(
    y_test.numpy(),
    test_predictions.numpy()
)

print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy:.2%}"
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "iris_model.pth"
)


# ============================================================
# 14. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "iris_scaler.pkl"
)


print("\nModel saved as: iris_model.pth")
print("Scaler saved as: iris_scaler.pkl")