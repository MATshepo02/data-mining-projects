# ITDAA4-12 Project — Data Mining and Data Administration

**Module:** ITDAA4-12 | NQF Level 8  
**Author:** Matshepo Tshabangu  
**Institution:** University of the Witwatersrand  
**Languages:** Python · PostgreSQL  

---

## Overview

This repository contains Python and PostgreSQL implementations for the ITDAA4-12 Data Mining and Data Administration Project. The project covers three distinct real-world applications:

- **Q1:** LDA Topic Modelling on 29 years of ICML research papers (1987–2016)
- **Q2:** Decision Tree Regression to predict asteroid diameter from NASA JPL data
- **Q3:** PostgreSQL database design and querying for a book store orders system

---

## Project Structure

```
├── Question1_LDA_TopicModelling.py      # LDA on ICML papers (35 marks)
├── Question2_DecisionTree_Asteroid.py   # Decision Tree regression (35 marks)
├── Question3_PostgreSQL_BookStore.sql   # Pure SQL script (30 marks)
├── Question3_PostgreSQL_Runner.py       # Python + psycopg2 runner for Q3
├── README.md
└── data/
    ├── articles.csv                     # ICML papers dataset (Q1)
    └── asteroid.csv                     # NASA asteroid dataset (Q2)
```

---

## Questions Summary

### Question 1 — LDA Topic Modelling on ICML Papers (35 Marks)
**File:** `Question1_LDA_TopicModelling.py`  
**Dataset:** `articles.csv` (ICML papers, 1987–2016)

| Section | Description | Marks |
|---|---|---|
| 1.1 Data cleaning | Missing values, duplicates, text sanitisation | 3 |
| 1.1 Exploratory analysis | Papers per year, word count, top words | 3 |
| 1.1 Text preprocessing | Stopword removal, lemmatisation, BoW corpus | 5 |
| 1.1 Optimal parameters | Coherence score grid search over k=3–15 | 5 |
| 1.1 LDA model training | Final model with optimal k, topic-document assignment | 5 |
| 1.1 Visualisations | Word clouds, keyword bars, pyLDAvis interactive HTML | 5 |
| 1.2 Report | Topic naming, interpretation, and thematic insights | 5 |
| 1.3 Trend analysis | Topic prevalence over time, stacked area chart, discussion | 4 |

**Outputs:** 9 plots + `lda_vis.html` interactive topic explorer

---

### Question 2 — Asteroid Diameter Prediction (35 Marks)
**File:** `Question2_DecisionTree_Asteroid.py`  
**Dataset:** `asteroid.csv` (NASA JPL Small-Body Database)

| Section | Description | Marks |
|---|---|---|
| Data cleaning | Missing value treatment, duplicate removal, outlier filtering | 5 |
| Categorical encoding | Label encoding with high-cardinality column handling | 3 |
| Feature selection | Mutual Information ranking + correlation heatmap | 5 |
| Feature plot | MI bar chart + correlation heatmap | 5 |
| Feature scaling | StandardScaler | 4 |
| Train/test split | 80/20 stratified split | 2 |
| Model fitting | DecisionTreeRegressor + max_depth CV tuning | 4 |
| Evaluation | MAE, RMSE, R², MAPE on train + test; actual vs predicted scatter | 3 |
| Tree plot | Graphical decision tree (first 4 levels) | 4 |

**Outputs:** 7 plots including feature importance, depth tuning, actual vs predicted, residuals, and decision tree visualisation

---

### Question 3 — Book Store Orders Database — PostgreSQL (30 Marks)
**File:** `Question3_PostgreSQL_BookStore.sql`  
**Python runner:** `Question3_PostgreSQL_Runner.py`

| Sub-question | Description | Marks |
|---|---|---|
| 3.1 | Create `book_store` database | 2 |
| 3.2 | Create `Orders` table + insert 4 sample rows | 7 |
| 3.3 | Filter: Total_Amount > 1000 AND Quantity_Ordered >= 2 | 3 |
| 3.4 | Insert Order 505 | 3 |
| 3.5 | Create `Customers` table + insert 3 records + FK constraint | 5 |
| 3.6 | JOIN query — all orders for Customer_ID = 11 | 4 |
| 3.7 | Aggregate — total books and amount per customer | 3 |
| 3.8 | Large orders (qty > 2) JOIN query | 3 |

---

## Setup & Requirements

### Python Packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn gensim nltk pyLDAvis wordcloud psycopg2-binary
```

**requirements.txt:**
```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.1
gensim>=4.3
nltk>=3.8
pyLDAvis>=3.4
wordcloud>=1.9
psycopg2-binary>=2.9
```

### PostgreSQL Setup (for Q3)
- Install PostgreSQL (v13+)
- Run `Question3_PostgreSQL_BookStore.sql` in psql or pgAdmin  
  **or** update credentials in `Question3_PostgreSQL_Runner.py` and run it with Python

---

## How to Run

```bash
# Question 1 — LDA Topic Modelling
# Place articles.csv in the same directory
python Question1_LDA_TopicModelling.py

# Question 2 — Decision Tree (Asteroid)
# Place asteroid.csv in the same directory
python Question2_DecisionTree_Asteroid.py

# Question 3 — PostgreSQL (SQL file)
psql -U postgres -f Question3_PostgreSQL_BookStore.sql

# Question 3 — PostgreSQL (Python runner)
# Update DB_CONFIG credentials first
python Question3_PostgreSQL_Runner.py
```

---

## Key Results

| Question | Technique | Key Output |
|---|---|---|
| Q1 | LDA Topic Modelling | Optimal k topics via coherence, topic trends 1987–2016 |
| Q2 | Decision Tree Regression | R², RMSE, MAE on asteroid diameter test set |
| Q3 | PostgreSQL | Normalised schema with FK, aggregate queries, JOINs |

---

## Technologies Used

- **Languages:** Python 3, PostgreSQL
- **Libraries:** gensim (LDA), nltk, pyLDAvis, scikit-learn, pandas, matplotlib, seaborn, psycopg2
- **Techniques:** LDA Topic Modelling, Decision Tree Regression, Mutual Information Feature Selection, SQL Database Design

---

## Academic Integrity

This project was completed individually in accordance with academic integrity policies. All code was written and understood by the author. Submitted through Turnitin in compliance with institutional requirements.

---

*Eduvos | BSc IT (Hons) Data Science | 2026*
