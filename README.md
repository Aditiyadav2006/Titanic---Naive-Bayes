# 🚢 Titanic Survival Prediction: Naive Bayes Classification

## Project Overview

This project focuses on building a machine learning model to predict the survival of passengers on the RMS Titanic. We utilize the **Gaussian Naive Bayes** algorithm, a simple yet powerful probabilistic classifier, for this binary classification task.

The project involves data preprocessing, feature engineering based on insights from initial Exploratory Data Analysis (EDA), and training/evaluating the Naive Bayes model.

## Key Files

| File Name | Description |
| :--- | :--- |
| `titanic_nb.ipynb` | The main Jupyter Notebook containing all the steps for data preparation, model training, evaluation, and results using the Gaussian Naive Bayes classifier. |
| `train.csv` | The primary dataset used for training and testing the model, containing passenger features and the 'Survived' target variable. |
| `README.md` | This overview file. |

## Methodology

### 1. Data Preprocessing & Feature Engineering
The initial data required cleaning and transformation before being fed into the model. The notebook covers:

* **Handling Missing Values:** Imputing missing values for `Age`, `Fare`, and `Embarked`.
* **Feature Transformation:** Converting categorical features (like `Sex` and `Embarked`) into numerical representations (e.g., using one-hot encoding or label encoding).
* **Feature Selection:** Dropping irrelevant or high-cardinality columns that are not useful for a Naive Bayes model (e.g., `Name`, `Ticket`, `Cabin`).
* **Data Splitting:** Dividing the dataset into training and testing sets to ensure robust model evaluation.

### 2. Gaussian Naive Bayes Model
The Gaussian Naive Bayes algorithm was chosen because it assumes that features follow a normal (Gaussian) distribution, which often performs well on datasets like the Titanic where features like `Age` and `Fare` can be approximated by a normal distribution.

* **Model Training:** The classifier is trained on the processed training data.
* **Prediction:** The trained model is used to make predictions on the unseen test set.

## Model Performance

The performance of the model is evaluated using standard classification metrics.

| Metric | Result (Example Placeholder) |
| :--- | :--- |
| **Accuracy** | 0.78 |
| **Precision** | 0.78 |
| **Recall** | 0.78 |
| **F1-Score** | 0.78 |

## Technologies and Libraries

This project is implemented in Python and utilizes the following libraries:

* **Python 3.x**
* **VS code** (for the analysis environment)
* `pandas` & `numpy` (for data manipulation)
* `matplotlib` & `seaborn` (for visualization of results)
* `scikit-learn` (for the Gaussian Naive Bayes classifier, `train_test_split`, and metrics like `accuracy_score`, `confusion_matrix`, and `classification_report`).

You can also view the notebook directly on GitHub or platforms like **Google Colab** without needing a local setup.

## 👩‍💻 Author
**Aditi K**  
CSE (AI & ML) | Titanic | NB
