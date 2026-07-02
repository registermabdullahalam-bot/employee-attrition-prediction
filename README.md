# Employee Attrition Prediction System

Machine learning project to predict employee attrition using the IBM HR Analytics dataset.

## Objective

Predict whether an employee is likely to leave the company using classification models, and translate the findings into actionable HR retention recommendations.

## Dataset

- **Source:** IBM HR Analytics Employee Attrition Dataset
- **Size:** 1,470 employee records, 35 features
- **Target:** `Attrition` (Yes/No) - 16.12% attrition rate (imbalanced classification problem)

## Workflow

1. **Data Loading & Exploration** - shape, types, target distribution
2. **Data Cleaning & Preprocessing** - no missing values or duplicates found; dropped non-informative columns (`EmployeeNumber`, `Over18`, `StandardHours`); encoded target; built a `ColumnTransformer` pipeline (StandardScaler for numeric, OneHotEncoder for categorical)
3. **Exploratory Data Analysis** - attrition by department, job role, income, work-life balance, tenure, and distance from home
4. **Model Building** - trained and compared 3 classifiers with `class_weight='balanced'`:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
5. **Model Evaluation** - Precision, Recall, F1-score, ROC-AUC, confusion matrices, ROC curve comparison
6. **Feature Importance** - extracted top drivers of attrition from Logistic Regression coefficients
7. **HR Insights & Recommendations** - business-facing conclusions from the analysis

## Results

| Model | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.349 | 0.638 | **0.451** | 0.803 |
| Random Forest | 0.467 | 0.298 | 0.364 | 0.797 |
| Gradient Boosting | **0.714** | 0.213 | 0.328 | **0.807** |

**Logistic Regression** was selected as the best overall model - it achieves the highest Recall and F1-score, meaning it's most effective at catching employees who are actually at risk of leaving, which matters more than raw accuracy in an imbalanced HR retention context.

## Key Findings

- **Sales Representatives** and **Laboratory Technicians** have the highest attrition risk
- Employees who **travel frequently** and **work overtime** are significantly more likely to leave
- **Work-life balance** and **income** matter, but neither alone explains attrition
- Most departures happen within an employee's **first 5 years**

## HR Recommendations

1. Introduce targeted retention programs (mentorship, career pathways, recognition) for Sales Representatives and Lab Technicians
2. Reduce excessive overtime and improve work-life balance through workload review and flexible arrangements

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn`

## Limitations

This model is based on historical data and doesn't capture personal circumstances, culture, or external economic factors. It should be used as a decision-support tool, not a sole basis for HR decisions.

## Repository Structure

```
├── analysis.ipynb          # Full analysis notebook
├── HR_Attrition.csv        # Dataset
├── charts/                 # Generated visualizations
└── README.md
```

---
*Part of an ongoing ML project series documenting my Data Science journey — follow along on [www.linkedin.com/in/mohammad-abdullah-alam].*
