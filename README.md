# Wine Classification with Neural Network

  

This project uses a multilayer neural network (MLP) developed with **TensorFlow and Keras** to classify wine types based on their chemical properties. The dataset used is the classic **Wine dataset** from scikit-learn.

  

## 📌 Technologies

  

- Python

- TensorFlow / Keras

- scikit-learn

- Matplotlib and Seaborn

  

## ⚙️ Project flow

  

- Data loading and normalization (`MinMaxScaler`)

- Division into training and testing

- One-Hot coding of labels

- Definition and training of a neural network

- Evaluation with accuracy, confusion matrix and ROC curves

  

## 🧠 Model structure

  

- 2 hidden layers (64 and 32 neurons, ReLU + Dropout)

- Softmax output layer (3 classes)

  

## 📊 Visualizations

  

- Confusion matrix

- ROC curves per class

- Loss and accuracy curves per epoch

  

## 📊 How to execute


```sh
pip install numpy matplotlib seaborn scikit-learn tensorflow
python wine_classification_mlp.py
```

  
