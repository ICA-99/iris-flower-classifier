import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered"
)


# ============================================================
# MODEL DEFINITION
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


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = IrisANN()

    model.load_state_dict(
        torch.load(
            "iris_model.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    return model


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    return joblib.load(
        "iris_scaler.pkl"
    )


model = load_model()
scaler = load_scaler()


# ============================================================
# CLASS MAPPING
# ============================================================

class_names = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}


# ============================================================
# UI
# ============================================================

st.title("🌸 Iris Flower Classifier")

st.write(
    "Enter the measurements of an Iris flower "
    "to predict its species."
)


# ============================================================
# INPUTS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        value=5.1,
        step=0.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        value=3.5,
        step=0.1
    )


with col2:

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        value=1.4,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        value=0.2,
        step=0.1
    )


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict 🌸",
    type="primary"
):

    # --------------------------------------------------------
    # 1. Create NumPy input
    # --------------------------------------------------------

    X = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])


    # --------------------------------------------------------
    # 2. Apply SAME scaler from training
    # --------------------------------------------------------

    X_scaled = scaler.transform(X)


    # --------------------------------------------------------
    # 3. Convert to PyTorch tensor
    # --------------------------------------------------------

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # 4. Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(X_tensor)


        # ----------------------------------------------------
        # Find highest output neuron
        # ----------------------------------------------------

        predicted_class = torch.argmax(
            outputs,
            dim=1
        ).item()


    # --------------------------------------------------------
    # 5. Convert class ID → species name
    # --------------------------------------------------------

    predicted_species = class_names[
        predicted_class
    ]


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.success(
        f"Predicted Species: **{predicted_species}**"
    )

    st.write(
        f"Class ID: **{predicted_class}**"
    )


    # ========================================================
    # SHOW MODEL OUTPUTS
    # ========================================================

    st.subheader("Model Output")

    st.write(
        "The ANN produces 3 logits, one for each class."
    )

    output_values = outputs[0].numpy()

    st.write({
        "Iris-setosa": float(output_values[0]),
        "Iris-versicolor": float(output_values[1]),
        "Iris-virginica": float(output_values[2])
    })