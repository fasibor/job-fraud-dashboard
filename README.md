# 🚨 Fake Job Fraud Intelligence Dashboard

A comprehensive **interactive dashboard** built with **Streamlit** for analyzing, visualizing, and predicting fake job postings. This project leverages **machine learning**, **text analytics**, and **geospatial insights** to help identify high-risk job postings.



## 🔹 Features

- **Executive Overview**
  - Key metrics: Total jobs, fraud rate, high-risk postings, and average fraud probability.
  - Top 10 high-risk jobs with fraud probability percentages.
  - Interactive bar chart of Real vs Fake job postings.

- **EDA Insights**
  - Text length distribution vs fraud.
  - Fraud probability histogram.
  - Employment type, experience level, and education level analysis with top 10 fraud categories.
  - Fraud detection thresholds visualization.
  - Word cloud of job descriptions.

- **Global Fraud Map**
  - Interactive geospatial bubble map showing fraud hotspots by country.

- **Text Intelligence**
  - Analyze text similarity using TF-IDF and cosine similarity.
  - Detect templated fraud patterns.

- **Predict Job**
  - Input job title and description to predict fraud risk.
  - Displays probability as HIGH or LOW risk.

- **Interactive Filters**
  - Employment type, education, experience level, country.
  - Fraud status and fraud probability sliders.
  - All charts update dynamically based on filter selections.



## 🔹 Tech Stack

- Python 3.x
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Plotly
- Matplotlib & WordCloud
- Joblib for ML model persistence
- Pycountry for location normalization



## 🔹 Setup & Run Locally

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <your-project-folder>
