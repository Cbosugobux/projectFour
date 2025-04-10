# Project Four-Team Four (NCCA Match Predictor & Visual Analytics Dashboards )

**Project Overview:**

This project machine learning and interactive visualizations to analyze NCAA Men's Basketball team performance and predict outcomes in the 2025 tournament. Our primary focus was on:
1. Predict which teams will make the Sweet 16
2. Predict the winner of individual NCAA tournament matchups. 

**Project Goals:**
  **Model 1:** Predict Sweet 16 teams based on team performance data
  **Model 2:** Predict game outcomes for NCAA matchups using team stats

Built by Team 4 in April 2025, this project combines web scraping, data analysis, machine learning, and interactive Flask web applications to simulate predictive sports analytics.

## **Summary**

### **Data Collection & Cleaning**
  - Use BeautifulSoup and python to scrape data from basketball-reference.com
  - Cleaned and merged data from multiple sources (team stats, game logs, rolling averages)

### **Model Training 1 : Predicting Sweet 16 Status**
  - Features like W-L%, SRS, SOS, AST, TOV, FG%, FT% were selected
  - Data scaled using RobustScaler, correlated features dropped using a correlation matrix.
  - Logistic Regression was used to predict Sweet 16 status

### **Model Training 2 : Game Score Prediction**
  - XGBoost Regressor with GridSearchCV for hyperparameter tuning was used to predict game scores. 
  - The metrics were Train/Test split evaluation and cross-validation score 
  - Added SRS adjustment to improve accuracy based on strength of schedule.
- Computed residual sigma for confidence estimation.
- Exported model and scaler with joblib for integration with the app.

### **Statistical Visualization**
  - Using Pandas, Numpy, and hvplot, we created visualizations comparing performance metrics such as Win %, SRS, SOS, AST, TOv, FG%, and FT% to Sweet 16 outcomes.
  - An interactive radar chart allows users to view and compare the overall profile of each team.
  - Bubble charts and animated charts display trends and highlight outliers.

  ### **Web App with Flask**
  - Developed a user-friendly Flask web application to access predictions.
  - Included a dropdown interface to select two teams to compare, outputing predicted scores, win margin, probability, and confidence level of matchup.
  - The backend supports two key endpoints: /train to retrain the model with updated data, and /predict to generate real-time game outcome predictions.

## **Technologies Used**
- NumPy 
- Pandas: data cleaning and analyzing 
- Scikit-learn (Sklearn)
- XGBoost: building regression models
- SciPy
- Matplotlib: create static charts like radar plots 
- Flask : backend API
- HTML : frontend dashboard or web interface.
- Webbrowser
- Threading – Flask server
- Google Colab


## **Contributors**
- **Kaouther Abid** 
- **Sara Bendahmane**
- **Chris Bushelman**
- **Chantelle Cane**
- **Melisa Hodzic** 
- **Myatminn Khant** 
- **Rosy Mathew** 
- **Amy Millimen-Tola** 
- **Sae Park** 
- **Matt Shea** 
- **Xavier Walsh**

## Data
https://www.basketball-reference.com/

