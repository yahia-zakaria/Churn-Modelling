# Churn Modelling & Salary Regression

A machine learning project that predicts customer churn and estimates salary using deep neural networks. The project includes two main models: a binary classification model for churn prediction and a regression model for salary estimation.

## 🎯 Project Overview

This project demonstrates end-to-end machine learning workflows including:
- **Customer Churn Prediction**: Binary classification model to predict whether a customer will leave the bank
- **Salary Regression**: Regression model to predict estimated salary based on customer features
- **Interactive Web Application**: Streamlit app for real-time churn predictions

## ✨ Features

- **Deep Neural Networks**: TensorFlow/Keras models with optimized architectures
- **GPU Acceleration**: Optimized for Apple Silicon (M4 Pro) with Metal Performance Shaders
- **Hyperparameter Tuning**: GridSearchCV for optimal model parameters
- **Data Preprocessing**: Label encoding, one-hot encoding, and feature scaling
- **Model Persistence**: Saved models and preprocessors for deployment
- **Visualization**: TensorBoard integration for training monitoring
- **Interactive UI**: Streamlit web application for predictions

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow 2.18.0, Keras 3.13.1
- **GPU Support**: tensorflow-metal 1.2.0 (Apple Silicon)
- **Machine Learning**: scikit-learn, scikeras
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, TensorBoard
- **Web Framework**: Streamlit
- **Development**: Jupyter Notebook, Python 3.12

## 📋 Prerequisites

- Python 3.12 (recommended for tensorflow-metal compatibility)
- macOS 12.0+ (for Apple Silicon GPU support)
- Conda or Miniconda

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yahia-zakaria/Churn-Modelling.git
cd Churn-Modelling
```

### 2. Create Conda Environment

```bash
conda create -n churn_gpu python=3.12 -y
conda activate churn_gpu
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify GPU Support (Optional)

```python
import tensorflow as tf
print("GPU devices:", tf.config.list_physical_devices('GPU'))
# Should output: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 📁 Project Structure

```
Churn-Modelling/
├── app.py                      # Streamlit web application
├── experiments.ipynb           # Churn prediction model development
├── salary_regression.ipynb      # Salary regression model development
├── predication.ipynb           # Prediction examples
├── Churn_Modelling.csv         # Dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── model.h5                    # Trained churn prediction model
├── salary_regression_model.keras # Trained salary regression model
│
├── label_encoder_gender.pkl    # Gender label encoder
├── oneHot_encoder_geo.pkl      # Geography one-hot encoder
├── scalar.pkl                  # Feature scaler
│
└── logs/                       # TensorBoard logs
    └── fit*/
```

## 🎓 Model Details

### Churn Prediction Model

- **Type**: Binary Classification
- **Architecture**: 
  - Input Layer: 12 features
  - Hidden Layer 1: 64 neurons (ReLU)
  - Hidden Layer 2: 32 neurons (ReLU)
  - Output Layer: 1 neuron (Sigmoid)
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Total Parameters**: 2,945

### Salary Regression Model

- **Type**: Regression
- **Architecture**:
  - Input Layer: 12 features
  - Hidden Layer 1: 64 neurons (ReLU)
  - Hidden Layer 2: 32 neurons (ReLU)
  - Output Layer: 1 neuron (Linear)
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Optimizer**: Adam

## 💻 Usage

### Running the Streamlit App

```bash
conda activate churn_gpu
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Training Models

Open the Jupyter notebooks to explore and train models:

```bash
conda activate churn_gpu
jupyter notebook
```

Then open:
- `experiments.ipynb` for churn prediction
- `salary_regression.ipynb` for salary regression

### Viewing TensorBoard Logs

```bash
conda activate churn_gpu
tensorboard --logdir logs
```

Or in Jupyter:
```python
%load_ext tensorboard
%tensorboard --logdir logs
```

## 🔧 Hyperparameter Tuning

The project includes GridSearchCV for hyperparameter optimization:

- **Neurons**: [16, 32, 64, 128]
- **Layers**: [1, 2]
- **Epochs**: [50, 100]

See `experiments.ipynb` for implementation details.

## 📊 Dataset

The dataset (`Churn_Modelling.csv`) contains customer information including:
- Credit Score
- Geography (France, Spain, Germany)
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Member Status
- Estimated Salary
- Exited (Target for churn prediction)

## 🎨 Features

### Data Preprocessing
- Label encoding for categorical variables (Gender)
- One-hot encoding for Geography
- Standard scaling for numerical features
- Train-test split (80/20)

### Model Training
- Early stopping to prevent overfitting
- TensorBoard callbacks for monitoring
- Model checkpointing
- GPU acceleration support

### Deployment
- Model serialization (.h5, .keras)
- Preprocessor persistence (.pkl)
- Streamlit web interface
- Real-time predictions

## 🖥️ GPU Acceleration

This project is optimized for Apple Silicon Macs (M1/M2/M3/M4):

- **GPU**: Uses Metal Performance Shaders via tensorflow-metal
- **Performance**: 3-10x faster training compared to CPU-only
- **Requirements**: macOS 12.0+, Python 3.9-3.12, tensorflow-metal 1.2.0

## 📝 Notes

- Model files (`.h5`, `.keras`, `.pkl`) are excluded from git via `.gitignore`
- Logs and TensorBoard outputs are also excluded
- The dataset should be placed in the project root directory

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Yahia Zakaria**
- GitHub: [@yahia-zakaria](https://github.com/yahia-zakaria)
- Email: yahiazakaria91@hotmail.com

## 🙏 Acknowledgments

- TensorFlow team for excellent deep learning framework
- Apple for Metal GPU acceleration support
- Streamlit for easy web app deployment
