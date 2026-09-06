# 🌸 Iris Flower Classifier

A simple end-to-end **multiclass classification project** built with
**PyTorch** and deployed using **Streamlit**.

The project trains an Artificial Neural Network (ANN) to classify Iris
flowers into three species based on four measurements:

-   Sepal Length
-   Sepal Width
-   Petal Length
-   Petal Width

The trained model and preprocessing scaler are saved and then loaded by
a Streamlit application for real-time predictions.

------------------------------------------------------------------------

## 🚀 Project Overview

The complete machine learning pipeline is:

``` text
Iris Dataset
     ↓
Feature Selection
     ↓
Train / Validation / Test Split
     ↓
StandardScaler
     ↓
PyTorch Tensors
     ↓
DataLoader
     ↓
ANN Training
     ↓
CrossEntropyLoss
     ↓
Adam Optimizer
     ↓
Evaluate Model
     ↓
Save Model + Scaler
     ↓
Streamlit Application
     ↓
User Input
     ↓
Scale Input
     ↓
ANN Prediction
     ↓
Argmax
     ↓
Predicted Iris Species
```

------------------------------------------------------------------------

## 🎯 Classes

The target classes are encoded as:

    Class ID Species
  ---------- -----------------
         `0` Iris-setosa
         `1` Iris-versicolor
         `2` Iris-virginica

The neural network therefore has **3 output neurons**, one for each
class.

------------------------------------------------------------------------

## 🧠 Model Architecture

The project uses a small fully connected Artificial Neural Network:

``` text
Input
4 features
   │
   ▼
Linear(4 → 16)
   │
   ▼
ReLU
   │
   ▼
Linear(16 → 8)
   │
   ▼
ReLU
   │
   ▼
Linear(8 → 3)
   │
   ▼
3 Class Logits
```

### Output layer

The final layer is:

``` python
nn.Linear(8, 3)
```

It produces three raw values called **logits**:

``` text
Class 0 score
Class 1 score
Class 2 score
```

The predicted class is selected using:

``` python
torch.argmax(outputs, dim=1)
```

The model does **not** use Softmax in the network because
`CrossEntropyLoss` is used during training.

Softmax can optionally be applied during inference when probabilities
need to be displayed.

------------------------------------------------------------------------

## 📊 Dataset

The project uses the classic Iris dataset.

Each sample contains four numerical features:

``` text
SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm
```

The target column is:

``` text
Species
```

The original species names are converted into numerical class IDs:

``` python
class_mapping = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}
```

------------------------------------------------------------------------

## 🗂️ Project Structure

``` text
iris_classifier/
│
├── Iris.csv
├── train.py
├── app.py
├── requirements.txt
├── README.md
│
├── iris_model.pth
└── iris_scaler.pkl
```

### Files

  File                 Purpose
  -------------------- ---------------------------------
  `Iris.csv`           Iris dataset
  `train.py`           Model training and evaluation
  `app.py`             Streamlit inference application
  `requirements.txt`   Python dependencies
  `iris_model.pth`     Saved PyTorch model weights
  `iris_scaler.pkl`    Saved `StandardScaler`
  `README.md`          Project documentation

------------------------------------------------------------------------

## ⚙️ Data Preprocessing

### 1. Train/Validation/Test Split

The dataset is divided into:

``` text
80% Training
10% Validation
10% Test
```

The split uses:

``` python
stratify=y
```

This keeps the class distribution approximately consistent across the
different datasets.

### 2. Standardization

The features are standardized using:

``` python
StandardScaler()
```

The scaler is fitted **only on the training data**:

``` python
X_train = scaler.fit_transform(X_train)
```

Validation and test data use the same fitted scaler:

``` python
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```

This prevents information from the validation/test sets from leaking
into preprocessing.

The trained scaler is saved as:

``` text
iris_scaler.pkl
```

This is important because the Streamlit application must apply the
**same preprocessing** to new user inputs.

------------------------------------------------------------------------

## 🏋️ Training

The model is trained using:

### Loss Function

``` python
nn.CrossEntropyLoss()
```

This is appropriate for the three-class classification problem.

### Optimizer

``` python
torch.optim.Adam(
    model.parameters(),
    lr=0.01
)
```

### Batch Size

``` text
16
```

### Epochs

``` text
100
```

------------------------------------------------------------------------

## 💾 Model Saving

After training, the learned model parameters are saved:

``` python
torch.save(
    model.state_dict(),
    "iris_model.pth"
)
```

The fitted scaler is saved using Joblib:

``` python
joblib.dump(
    scaler,
    "iris_scaler.pkl"
)
```

The two files are required for inference:

``` text
iris_model.pth
iris_scaler.pkl
```

------------------------------------------------------------------------

## 🔮 Inference Process

When a user enters flower measurements in the Streamlit application:

``` text
User Input
   │
   ▼
NumPy Array
   │
   ▼
Saved StandardScaler
   │
   ▼
Scaled Features
   │
   ▼
PyTorch Tensor
   │
   ▼
ANN
   │
   ▼
3 Logits
   │
   ▼
Argmax
   │
   ▼
Class ID
   │
   ▼
Class Name
```

For example, the model might produce:

``` text
[8.17, -3.90, -21.39]
```

The largest value is:

``` text
8.17
```

which belongs to class `0`.

Therefore:

``` text
0 → Iris-setosa
```

------------------------------------------------------------------------

## 🖥️ Streamlit Application

The Streamlit application provides four input fields:

``` text
Sepal Length
Sepal Width
Petal Length
Petal Width
```

After clicking **Predict**, the application:

1.  Reads the user's measurements.
2.  Applies the saved `StandardScaler`.
3.  Converts the values into a PyTorch tensor.
4.  Passes the tensor through the trained ANN.
5.  Gets three output logits.
6.  Uses `argmax` to select the highest-scoring class.
7.  Converts the class ID into the species name.
8.  Displays the prediction.

The application also displays the raw model logits for educational
purposes.

------------------------------------------------------------------------

## 🛠️ Installation

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd iris_classifier
```

### 2. Create a virtual environment

Linux/macOS:

``` bash
python -m venv venv
source venv/bin/activate
```

Windows:

``` bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📦 Requirements

Example `requirements.txt`:

``` text
pandas
numpy
torch
scikit-learn
joblib
streamlit
```

------------------------------------------------------------------------

## 🏋️ Train the Model

Run:

``` bash
python train.py
```

The training script will:

-   Load `Iris.csv`
-   Select the four input features
-   Encode the target classes
-   Split the data
-   Standardize the features
-   Train the ANN
-   Evaluate validation performance
-   Evaluate final test performance
-   Save the trained model
-   Save the scaler

After successful training, these files should exist:

``` text
iris_model.pth
iris_scaler.pkl
```

------------------------------------------------------------------------

## 🌐 Run the Streamlit Application

After training:

``` bash
streamlit run app.py
```

Streamlit will start the application and provide a local URL.

Open the displayed URL in your browser.

------------------------------------------------------------------------

## 📈 Example Prediction

Input:

``` text
Sepal Length: 5.1
Sepal Width:  3.5
Petal Length: 1.4
Petal Width:  0.2
```

Possible output:

``` text
Predicted Species: Iris-setosa
Class ID: 0
```

The model may produce logits similar to:

``` text
Iris-setosa       →  8.17
Iris-versicolor   → -3.90
Iris-virginica    → -21.39
```

The highest logit is associated with `Iris-setosa`, so the predicted
class is `0`.

------------------------------------------------------------------------

## 🔍 Why Softmax Is Not Used During Training

The final layer produces raw logits:

``` python
nn.Linear(8, 3)
```

There is intentionally no:

``` python
nn.Softmax()
```

because:

``` python
nn.CrossEntropyLoss()
```

expects raw logits and internally performs the appropriate
log-softmax/cross-entropy computation.

For class selection:

``` python
torch.argmax(outputs, dim=1)
```

is sufficient.

If probabilities are required for display:

``` python
probabilities = torch.softmax(outputs, dim=1)
```

can be applied during inference.

------------------------------------------------------------------------

## 🔐 Important Deployment Detail

The model and scaler must be treated as a pair:

``` text
iris_model.pth
       +
iris_scaler.pkl
```

The model was trained using standardized features. Therefore, production
inputs must be transformed using the **same fitted scaler**.

Do not create and fit a new scaler during prediction.

Incorrect:

``` python
new_scaler = StandardScaler()
new_scaler.fit_transform(user_input)
```

Correct:

``` python
scaler = joblib.load("iris_scaler.pkl")
scaled_input = scaler.transform(user_input)
```

------------------------------------------------------------------------

## 🧪 Model Evaluation

The test set is used only for the final evaluation.

Accuracy is calculated using:

``` python
accuracy_score(
    y_test.numpy(),
    test_predictions.numpy()
)
```

Because the Iris dataset is small and relatively easy to classify, this
ANN can typically achieve high test accuracy, although the exact result
can vary slightly depending on the training configuration.

------------------------------------------------------------------------

## 🚀 Deployment

This project uses **Streamlit** for the user-facing deployment.

For local deployment:

``` bash
streamlit run app.py
```

For cloud deployment, the project can be deployed to a
Streamlit-compatible hosting platform with:

``` text
Iris.csv
app.py
requirements.txt
iris_model.pth
iris_scaler.pkl
```

The training script does not need to run every time the application
starts. The deployed application loads the already-trained model and
scaler.

------------------------------------------------------------------------

## ⚠️ Production Considerations

This project is designed as a clean educational/demo deployment. For a
larger production system, consider adding:

-   Input validation and sensible feature ranges
-   Dependency version pinning
-   Model/version tracking
-   Logging
-   Error handling
-   Health checks
-   Automated testing
-   Containerization with Docker
-   CI/CD
-   Model monitoring
-   Secure artifact management

------------------------------------------------------------------------

## 🧰 Technologies Used

-   **Python**
-   **Pandas** --- data loading and manipulation
-   **NumPy** --- numerical processing
-   **Scikit-learn** --- data splitting, scaling, and evaluation
-   **PyTorch** --- ANN model and training
-   **Joblib** --- scaler serialization
-   **Streamlit** --- web application and deployment UI

------------------------------------------------------------------------

## 📚 Learning Objectives

This project demonstrates the complete lifecycle of a basic multiclass
deep-learning application:

``` text
Dataset
   ↓
Preprocessing
   ↓
Train/Test Split
   ↓
Standardization
   ↓
Tensor Conversion
   ↓
DataLoader
   ↓
ANN
   ↓
Forward Pass
   ↓
Cross Entropy Loss
   ↓
Backpropagation
   ↓
Adam Optimizer
   ↓
Validation
   ↓
Testing
   ↓
Model Serialization
   ↓
Inference
   ↓
Streamlit Deployment
```

It is particularly useful for understanding how a PyTorch classification
model moves from **training code to an actual user-facing application**.

------------------------------------------------------------------------

## 👨‍💻 Author

**CodeWizard**

Built as a PyTorch multiclass classification and Streamlit deployment
project.
