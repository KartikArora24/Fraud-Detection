# 💳 Credit Card Fraud Detection (Streamlit App)

Unsupervised anomaly detection web application built with **Streamlit**, **Isolation Forest**, and **Local Outlier Factor (LOF)**, calibrated at a **70% validation precision** target.

## Project Structure

```text
credit-card-fraud-app/
├── app.py                  # Streamlit web application
├── train_model.py          # One-time training & artifact generation script
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignores creditcard.csv and local caches
└── artifacts/
    └── fraud_artifacts.joblib  # Pre-trained models, scaler, and cutoffs
```

## Quickstart

### 1. Train the Model Locally
Place `creditcard.csv` in the project root folder (it is ignored by `.gitignore` so it won't block your GitHub push), then run:

```bash
pip install -r requirements.txt
python train_model.py
```

This generates `artifacts/fraud_artifacts.joblib`.

### 2. Run the App Locally
```bash
streamlit run app.py
```

### 3. Push to GitHub & Deploy on Streamlit Cloud
1. Initialize Git and push to your repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Credit card fraud detection app"
   git branch -M main
   git remote add origin <YOUR_GITHUB_REPO_URL>
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io), select your repository, set `app.py` as the main file path, and click **Deploy**.
