from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import joblib
import webbrowser
import threading

app = Flask(__name__)

# File paths for the CSV files.
FILE_PATH1 = r"Data\Gamelog_Averages_5.csv"
FILE_PATH2 = r"Data\03172025TeamBasicStats.csv"
MODEL_FILE = "optimal_model.pkl"

def train_and_save_model():
    # 1) Load CSVs
    df = pd.read_csv(FILE_PATH1)
    df2 = pd.read_csv(FILE_PATH2)

    # 2) Normalize merge keys
    df2.rename(columns={"School": "School Name"}, inplace=True)
    df["School Name"] = df["School Name"].astype(str).str.strip().str.lower()
    df2["School Name"] = df2["School Name"].astype(str).str.strip().str.lower()

    # 3) Merge SRS
    original_df = df.merge(df2[["School Name", "SRS"]], on="School Name", how="left")

    # 4) Target
    target = "Score Tm_adv"

    # 5) Drop highly correlated features
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    threshold = 0.9
    to_drop = [col for col in upper.columns if any(upper[col] > threshold) and col != target]
    df_reduced = df.drop(columns=to_drop)

    # 6) Prepare X, y
    X = df_reduced.drop(columns=[target, "School Name", "SRS"], errors='ignore')
    y = df_reduced[target]

    # 7) Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 8) Split (capture indices separately)
    idx = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, idx, test_size=0.4, random_state=42
    )

    # 9) Hyper‐param search
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }
    xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    grid_search = GridSearchCV(xgb_reg, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # 10) Evaluate
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring="r2")

    # 11) Full‐data preds & store
    original_df["Predicted_Scores"] = best_model.predict(X_scaled)

    # 12) Residual sigma
    sigma = np.std(y_test - best_model.predict(X_test))
    sigma = max(sigma, 1e-5)

    # 13) Optimize SRS weight
    avg_SRS = original_df["SRS"].mean()
    raw_test = best_model.predict(X_test)
    srs_test = original_df.loc[idx_test, "SRS"].values
    best_weight, best_rmse = 0.0, np.inf
    for w in np.linspace(0, 1, 11):
        adj = raw_test + w * (srs_test - avg_SRS)
        rmse = np.sqrt(mean_squared_error(y_test, adj))
        if rmse < best_rmse:
            best_rmse, best_weight = rmse, w

    # 14) Save everything
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "avg_SRS": avg_SRS,
        "srs_weight": best_weight,
        "sigma": sigma,
        "df_predictions": original_df
    }, MODEL_FILE)

    return {
        "train_score": train_score,
        "test_score": test_score,
        "cv_mean_score": float(np.mean(cv_scores)),
        "best_params": grid_search.best_params_,
        "srs_weight": best_weight,
        "sigma": sigma
    }


def get_team_info(team_name, df_predictions):
    match = df_predictions[df_predictions["School Name"] == team_name.lower()]
    if match.empty:
        return None, None
    return float(match["Predicted_Scores"].iloc[0]), float(match["SRS"].iloc[0])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():
    try:
        results = train_and_save_model()
        return jsonify({"status": "ok", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        t1, t2 = data.get("team1", "").strip(), data.get("team2", "").strip()
        if not t1 or not t2:
            return jsonify({"status": "error", "message": "Both team1 and team2 required"}), 400

        mo = joblib.load(MODEL_FILE)
        pred1, srs1 = get_team_info(t1, mo["df_predictions"])
        pred2, srs2 = get_team_info(t2, mo["df_predictions"])
        if pred1 is None or pred2 is None:
            missing = [n for n,p in [(t1,pred1),(t2,pred2)] if p is None]
            return jsonify({"status": "error", "message": f"Team(s) not found: {', '.join(missing)}"}), 404

        w, μ, σ = mo["srs_weight"], mo["avg_SRS"], mo["sigma"]
        adj1 = pred1 + w * (srs1 - μ)
        adj2 = pred2 + w * (srs2 - μ)
        margin = adj1 - adj2
        prob1 = norm.sf(0, loc=margin, scale=σ)
        outcome = "Team 1 wins" if margin>0 else "Team 2 wins" if margin<0 else "Tie"

        return jsonify({"status":"success","result":{"outcome": outcome,"probability": prob1}})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500


if __name__ == "__main__":
    # open browser after a short delay
    threading.Timer(1.25, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    app.run(debug=True, port=5000)
