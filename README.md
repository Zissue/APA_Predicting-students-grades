# ***README***

[Full Paper](https://github.com/Zissue/APA_Predicting-students-grades/blob/main/docs/Report_project_Leo_Zixuan.pdf) | [Data](https://github.com/Zissue/APA_Predicting-students-grades/blob/main/data/student-mat.csv) | [Code](https://github.com/Zissue/APA_Predicting-students-grades/blob/main/code/Project.ipynb)


![](https://i.imgur.com/sZ8MvdO.png)

**Authors**: 

*Leo Arriola, Zixuan Sun*

> **NOTE:** To install all the requirements from *requirements.txt*, please execute:  
> ```pip install -r requirements.txt```


In this project, we are going to study the correlation between demographic attributes and their relationship with academic performance. The dataset we've chosen is formed by attributes that a priori are interrelated with the academic performance of secondary school students. This includes the level of education of their parents, their parent's job and other similar attributes. The dataset allows us to treat it as a regression problem or through classification problem. In summary, the dataset has *33* attributes (some numerical and some categorical) and a total of *395* samples.


The dataset was extracted from the *UC Irvine Machine Learning Repository}*, it is a collection of databases, domain theories, and data generators that are used by the machine learning community for the empirical analysis of machine learning algorithms. It was originated by *David Aha* and fellow graduates from *UC Irvine*.


The full dataset can be obtained from [here](https://archive.ics.uci.edu/ml/datasets/student+performance), this includes the **.csv** files and other descriptive documents about the dataset. Regarding the source of this dataset, we know that it was collected through school reports and questionnaires from a Portuguese school (in *Cortez \& Silva*, *2008*).

---

# **FINAL APA PROJECT**  

## *Predicting Students' Grades*

### Abstract

This project aims to predict the final grades of secondary school students based on demographic and academic factors. Using machine learning, we examine correlations between students' backgrounds and their academic performance, specifically in mathematics. The dataset, sourced from the UC Irvine Machine Learning Repository, includes 33 attributes from family background to school life metrics, comprising 395 samples in total. We experimented with multiple regression models to find the most predictive model, concluding that a Random Forest Regressor provided the best performance. This README provides an overview of the project steps, models, and findings.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Description](#problem-description)
3. [Dataset](#dataset)
4. [Data Exploration & Preprocessing](#data-exploration--preprocessing)
5. [Model Selection and Training](#model-selection-and-training)
6. [Evaluation and Results](#evaluation-and-results)
7. [Conclusions](#conclusions)
8. [Future Work](#future-work)
9. [References](#references)

---

## 1. Introduction

In this project, we explore the relationships between students' demographic attributes (e.g., parents' education and occupation) and their academic performance in mathematics. By analyzing this dataset as a regression problem, we attempt to predict students’ final grades (G3) and identify significant factors that could influence academic outcomes. Through this analysis, we aim to understand how non-academic attributes impact students' performance and offer insights into potential interventions.

## 2. Problem Description

The dataset allows for a dual approach: either regression or classification. Here, we focus on regression, where the target variable is the final grade (`G3`) of the students in mathematics. Our goal is to predict this target grade based on the various features provided in the dataset. 

## 3. Dataset

The dataset is sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/student+performance) and consists of demographic and academic information collected through school reports and student questionnaires in Portugal. The dataset includes:
- **33 Attributes**: Mixture of categorical and numerical data.
- **395 Samples**: Data points representing individual students.
  
Attributes include student background details, family information, and academic performance metrics (e.g., `G1`, `G2`, intermediate grades, and `G3`, the final grade). This data enables the analysis of how factors like family structure, extracurricular activities, and parental education correlate with students' academic outcomes.

## 4. Data Exploration & Preprocessing

Data exploration and preprocessing were essential to preparing the dataset for modeling. This process included the following steps:

- **Missing Values**: Verified there were no missing values in the dataset.
- **Outliers Detection**: Detected outliers in features like `failures`, `famrel`, `absences`, and `age`. Retained them to maintain data integrity.
- **Encoding Categorical Variables**: Converted categorical features to numeric values using `pd.factorize`.
- **Feature Engineering**: Created a new feature, `performanceFailure`, based on past grades (`G1` and `G2`) and `failures` to capture historical academic performance in a single attribute.
- **Feature Selection**: Dropped highly correlated features (`G1` and `G2`) to reduce redundancy and improve model efficiency.
- **Normalization**: Applied Min-Max scaling to `age` and `absences` to standardize numerical values and reduce outlier influence.

## 5. Model Selection and Training

We experimented with various regression models, both linear and non-linear, to identify the best-performing model. Below are the models tested:

### Linear Models

1. **Linear Regression**: Used as a baseline model to establish initial performance.
2. **Ridge Regression**: Applied L2 regularization to handle potential overfitting.
3. **K-Nearest Neighbors (KNN)**: Implemented for localized predictions based on neighboring data points.

### Non-linear Models

1. **MLP Regressor (Neural Network)**: Tested for capturing complex patterns, though convergence issues limited its performance.
2. **Random Forest Regressor**: Selected as the best-performing model due to its high R² score, outperforming other models in generalization capability.

### Hyperparameter Tuning

For each model, we used GridSearchCV to optimize hyperparameters, testing settings such as regularization strength in Ridge, neighbor counts in KNN, and tree count/depth in Random Forest.

## 6. Evaluation and Results

The **Random Forest Regressor** was chosen as the final model based on its performance on test data, achieving an R² score of 0.83. This model’s ensemble approach (bagging of decision trees) enabled it to generalize effectively across test samples. 

### Key Metrics:

| Model                 | R² Train | R² Test | MAE (Test) |
|-----------------------|----------|---------|------------|
| Linear Regression     | 0.70     | 0.76    | 1.48       |
| Ridge Regression      | 0.69     | 0.74    | 1.58       |
| K-Nearest Neighbors   | 0.65     | 0.69    | 1.74       |
| MLP Regressor         | 0.69     | 0.73    | 1.60       |
| **Random Forest**     | **0.81** | **0.83**| **1.23**   |

## 7. Conclusions

The project demonstrates the feasibility of predicting student performance based on demographic and school-related factors. Notably, family background and previous academic metrics like `G1` and `G2` grades have substantial predictive power. The Random Forest model proved robust in generalization and was particularly effective in identifying key predictive features. This approach could be adapted to assist educators in early identification of students at risk of low performance.

## 8. Future Work

Opportunities for future improvements and extensions include:

- **Feature Expansion**: Further explore potential attributes, such as student engagement or additional family metrics, that may improve prediction accuracy.
- **Classification Models**: Implement classification approaches to predict grade ranges or performance tiers, offering an alternative perspective on academic performance.
- **Advanced Models**: Experiment with models like Gradient Boosting or deep learning frameworks for potential improvements in predictive accuracy.
  
---

## 9. References

- UC Irvine Machine Learning Repository: [Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Toward Data Science](https://towardsdatascience.com/)
