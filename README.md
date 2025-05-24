# 📈 Text Mining Project: Stock Sentiment - Predicting market behavior from tweets
This is our project repository for Text Mining, where we tackled the stock sentiment, predicting market behaviour from tweets. This repository includes everything from problem description and datasets to our implementation, experimental notebooks, and results.

## 🎯 Project Overview  
This project focuses on analyzing tweets related to the stock market and predicting the market sentiment conveyed by each tweet:  
- **Bearish (0):** Negative outlook on the market  
- **Bullish (1):** Positive outlook on the market  
- **Neutral (2):** Neither positive nor negative sentiment  

---

## 🗂 Repository Structure  

- `Project_Data/`  
  Contains the dataset files:  
  - `train.csv` (9543 tweets with sentiment labels)  
  - `test.csv` (299 tweets without labels; predictions required)  

- `Notebooks/`  
  - `tm_tests_42.ipynb` — Exploratory data analysis, preprocessing, feature engineering experiments, and multiple model tests.  
  - `tm_final_42.ipynb` — Final pipeline notebook with the best-performing model for prediction.

- `Predictions/`  
  - `pred_42.csv` — CSV file with predicted labels for the test set (columns: `id`, `predicted_label`).

- `Reports/`  
  - `report_42.pdf` — Project report documenting methods, experiments, and results.

---

## 🚀 Getting Started

1. Clone the repository.
2. Install any necessary dependencies listed in the notebooks.
3. Run the `tm_tests_42.ipynb` notebook to explore algorithm settings.
4. Use the `tm_final_42.ipynb` notebook to see final solution.

---

## 📂 Explore & Contribute

Feel free to explore the repository, run the notebooks, and use the code for your own experiments.  
We welcome feedback, suggestions, and contributions!