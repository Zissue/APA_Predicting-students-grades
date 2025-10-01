# Predicting Students' Grades

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Test Example Script](https://github.com/Zissue/APA_Predicting-students-grades/actions/workflows/test.yml/badge.svg)](https://github.com/Zissue/APA_Predicting-students-grades/actions/workflows/test.yml)

**Machine Learning Project for Academic Performance Analysis**

[Full Paper](https://github.com/Zissue/APA_Predicting-students-grades/blob/main/docs/Report_project_Leo_Zixuan.pdf) | [Data](https://github.com/Zissue/APA_Predicting-students-grades/blob/main/data/student-mat.csv) | [Code](https://github.com/Zissue/APA_Predicting-students-grades/blob/main/code/Project.ipynb)

![Project Banner](https://i.imgur.com/sZ8MvdO.png)

**Authors**: *Leo Arriola, Zixuan Sun*

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [How to Use](#how-to-use)
5. [Introduction](#introduction)
6. [Problem Description](#problem-description)
7. [Dataset](#dataset)
8. [Data Exploration & Preprocessing](#data-exploration--preprocessing)
9. [Model Selection and Training](#model-selection-and-training)
10. [Evaluation and Results](#evaluation-and-results)
11. [Conclusions](#conclusions)
12. [Future Work](#future-work)
13. [References](#references)
14. [Contributing](#contributing)
15. [License](#license)

---

## 🔍 Overview

This project aims to predict the final grades of secondary school students based on demographic and academic factors using machine learning techniques. We analyze correlations between students' backgrounds and their academic performance in mathematics.

**Key Highlights:**
- 📊 Dataset: 395 student samples with 33 attributes
- 🎯 Best Model: Random Forest Regressor (R² = 0.83)
- 📈 Mean Absolute Error: 1.23 grade points
- 🔬 Comprehensive exploratory data analysis and feature engineering

---

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Git

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Zissue/APA_Predicting-students-grades.git
   cd APA_Predicting-students-grades
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

5. **Open the project notebook:**
   Navigate to `code/Project.ipynb` in the Jupyter interface.

---

## 📁 Project Structure

```
APA_Predicting-students-grades/
├── code/
│   └── Project.ipynb          # Main analysis notebook
├── data/
│   └── student-mat.csv        # Student performance dataset
├── docs/
│   ├── Report_project_Leo_Zixuan.pdf  # Full project report
│   └── student.txt            # Dataset attribute descriptions
├── example.py                 # Quick demo script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
├── CITATION.cff               # Citation information
├── CONTRIBUTING.md            # Contribution guidelines
└── .gitignore                # Git ignore rules
```

---

## 🎯 How to Use

### Quick Start with Example Script

For a quick demonstration of the core functionality, run the example script:

```bash
python example.py
```

This will:
- Load and preprocess the student dataset
- Train a Random Forest model
- Display performance metrics and feature importance
- Complete in under a minute

### Full Analysis with Jupyter Notebook

1. **Run the complete analysis:**
   - Open `code/Project.ipynb` in Jupyter Notebook
   - Run all cells sequentially (Cell → Run All)
   - Review the outputs, visualizations, and model results

2. **Explore specific sections:**
   - **Data Preprocessing:** Cells covering data cleaning and feature engineering
   - **Exploratory Analysis:** Visualizations and statistical summaries
   - **Model Training:** Implementation of various regression models
   - **Model Evaluation:** Performance metrics and comparison

3. **Modify and experiment:**
   - Try different hyperparameters in the GridSearchCV sections
   - Add new features or preprocessing steps
   - Test alternative models

### Dataset

The dataset (`data/student-mat.csv`) contains information about Portuguese secondary school students, including:
- Demographic data (age, gender, family size)
- Family background (parents' education and occupation)
- School-related factors (study time, failures, support)
- Social factors (activities, relationships, alcohol consumption)
- Academic grades (G1, G2, G3)

For detailed attribute descriptions, see `docs/student.txt`.

---

## 📊 Introduction

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

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:
- Report bugs
- Suggest enhancements
- Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

If you use this project in your research or work, please cite it:

```bibtex
@software{arriola_sun_2024_student_grades,
  author = {Arriola, Leo and Sun, Zixuan},
  title = {Predicting Students' Grades using Machine Learning},
  year = {2024},
  url = {https://github.com/Zissue/APA_Predicting-students-grades},
  note = {A machine learning project that predicts secondary school students' final grades}
}
```

Alternatively, you can use the [CITATION.cff](CITATION.cff) file.

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the authors:
- Leo Arriola
- Zixuan Sun

---

**⭐ If you find this project helpful, please consider giving it a star!**
