# **Cardiac Disease Detection Using Neural Network And ANFIS-GA**

## **Table of Contents**
- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## **Overview**

This project focuses on the detection of cardiac disease using a combination of Neural Networks and Adaptive Neuro-Fuzzy Inference System (ANFIS) optimized with Genetic Algorithms (GA). The goal is to predict the risk of cardiac disease based on various health parameters such as:

- Age
- Cholesterol levels
- Heart rate
- Diabetes
- Family history
- Smoking
- Obesity
- Alcohol consumption
- Exercise
- Previous problems
- Medication
- Stress level
- Sedentary lifestyle
- BMI
- Triglycerides
- Physical activity
- Sleep hours

The project is implemented in Python and utilizes libraries such as **NumPy, Pandas, Scikit-learn, and Matplotlib** for data processing, model training, and visualization. The ANFIS-GA model is custom-built to optimize the fuzzy inference system using genetic algorithms for better prediction accuracy.

## **Project Objectives**

This project aims to develop a **Neural Network and ANFIS-GA-based classification model** to predict the risk of heart problems in individuals by leveraging relevant health data. Additionally, it is designed to train Neural Networks and ANFIS-GA to discern intricate patterns and nonlinear associations, enhancing the model's ability to predict the likelihood of heart problems with greater accuracy.

## **Project Structure**

The project is structured as follows:

1. **Data Preprocessing**:
   - Load dataset from a CSV file.
   - Normalize data using MinMaxScaler.
   - Split dataset into training and testing sets.

2. **ANFIS-GA Model**:
   - Implement ANFIS-GA model in a custom class `EVOLUTIONARY_ANFIS`.
   - Utilize Gaussian membership functions.
   - Optimize model using genetic algorithms.
   - Train and evaluate model.

3. **Model Training and Evaluation**:
   - Train ANFIS-GA model over multiple generations.
   - Evaluate performance using Root Mean Squared Error (RMSE).
   - Predict cardiac disease risk.

4. **Visualization**:
   - Plot actual vs. predicted risk values.

## **Requirements**

To run this project, you need the following Python libraries:

- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**

Install these libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## **Usage**

1. **Data Preparation**:
   - Ensure your dataset is in CSV format and contains necessary features.
   - Update the file path in the code to point to your dataset.

2. **Model Training**:
   - Initialize ANFIS-GA model with parameters.
   - Train the model using the `fit` method.

3. **Prediction**:
   - Use the trained model to predict risk.
   - Evaluate predictions.

4. **Visualization**:
   - Plot actual vs. predicted risk values.

## **Results**

- Model performance is evaluated using **RMSE**.
- **Actual vs. Predicted** risk values are visualized.
- ANFIS-GA optimization enhances accuracy.

## **Conclusion**

This project demonstrates the application of **Neural Networks** and **ANFIS-GA** for **cardiac disease detection**. The combination of these techniques allows for a more accurate and robust prediction model. Future enhancements can include:

- Additional feature engineering.
- Hyperparameter optimization.
- Exploring alternative machine learning models.

## **License**

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

## **Acknowledgments**

- The dataset used in this project is sourced from **[Source](https://www.kaggle.com/datasets?search=heart+)**.
- Special thanks to contributors and researchers whose work has been referenced.

---

