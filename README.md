# Crypto Wallet Credit Scoring Project

![Crypto Wallet](https://img.icons8.com/color/48/000000/bitcoin.png) ![Credit Score](https://img.icons8.com/color/48/000000/credit-score.png)

## 📖 Project Overview
This project develops a machine learning-based system to assign a "credit score" to cryptocurrency wallets. By analyzing on-chain transaction data, the system engineers various behavioral features and uses an XGBoost Regressor to predict a heuristic credit score, providing a quantitative measure of a wallet's activity and perceived reliability within a decentralized finance (DeFi) context.

The goal is to demonstrate how historical on-chain data can be leveraged to infer behavioral patterns and assign a score, which could potentially be used in various DeFi applications (e.g., risk assessment for lending protocols).

## ⚙️ Features
- 📥 **Data Ingestion**: Loads transaction data from a JSON file.
- ⚙️ **Feature Engineering**: Extracts meaningful features from raw transaction data, such as total transaction count, unique transaction types, time-based activity metrics, total deposited/borrowed/repaid values, net flows, and liquidation history.
- 📊 **Heuristic Scoring**: Implements a rule-based system to assign a "ground truth" credit score (0-1000) based on observed wallet behaviors (e.g., penalizing liquidations, rewarding repayments). This serves as the target for the ML model.
- 🏋️ **Model Training**: Trains an XGBoost Regressor model to learn the relationship between engineered features and the heuristic scores. Includes data splitting, feature scaling, and early stopping to prevent overfitting.
- 💾 **Model Persistence**: Saves the trained model and necessary preprocessing scalers for future use.
- 🏦 **Credit Score Generation**: Uses the trained model to predict credit scores for new (or all) wallets.
- 📤 **Results Export**: Exports the generated wallet scores to a CSV file.
- 📈 **Visualization**: Generates a histogram to visualize the distribution of credit scores.

## 📁 Project Structure
```
.
├── data/
│   └── transactions.json       # Input raw transaction data
├── src/
│   ├── models/                 # Directory to store trained models and scalers
│   │   ├── credit_score_model.pkl
│   │   ├── feature_scaler.pkl
│   │   ├── score_output_scaler.pkl
│   │   └── feature_columns.pkl
│   ├── train_model.py          # Script for training the credit scoring model
│   └── score_generator.py      # Script for generating credit scores using the trained model
├── wallet_credit_scores.csv    # Output CSV file with wallet addresses and their scores
├── score_distribution.png      # Output plot showing the distribution of credit scores
├── README.md                   # This README file
└── analyze.md                  # Detailed analysis of the project
```

## 🛠️ Setup and Installation
Follow these steps to set up and run the project:

### 📥 Clone the Repository (or create the project structure manually):
If you have a Git repository, clone it:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```
Otherwise, create the directory structure as shown above (data/, src/, src/models/).

### 🐍 Create a Virtual Environment:
It's highly recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv
```

### 🔄 Activate the Virtual Environment:
**Windows:**
```bash
.\venv\Scripts\activate
```
**macOS/Linux:**
```bash
source venv/bin/activate
```
Your terminal prompt should now show `(venv)` indicating the virtual environment is active.

### 📦 Install Dependencies:
Install the required Python libraries.
```bash
pip install pandas scikit-learn xgboost matplotlib seaborn joblib
```
**Note**: Ensure your XGBoost version is 2.1.0 or newer for compatibility with the provided scripts. Use `pip install --upgrade xgboost` to get the latest compatible version.

### 📂 Place Transaction Data:
Ensure your `transactions.json` file is placed inside the `data/` directory. This file should contain a list of transaction objects, similar to the provided sample:
```json
[
  {
    "_id": { "$oid": "681d38fed63812d4655f571a" },
    "userWallet": "0x00000000001accfa9cef68cf5371a23025b6d4b6",
    "network": "polygon",
    "protocol": "aave_v2",
    "txHash": "0x695c69acf608fbf5d38e48ca5535e118cc213a89e3d6d2e66e6b0e3b2e8d4190",
    "logId": "0x695c69acf608fbf5d38e48ca5535e118cc213a89e3d6d2e66e6b0e3b2e8d4190_Deposit",
    "timestamp": 1629178166,
    "blockNumber": 1629178166,
    "action": "deposit",
    "actionData": {
      "type": "Deposit",
      "amount": "2000000000",
      "assetSymbol": "USDC",
      "assetPriceUSD": "0.9938318274296357543568636362026045",
      "poolId": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
      "userId": "0x00000000001accfa9cef68cf5371a23025b6d4b6"
    },
    "__v": 0,
    "createdAt": { "$date": "2025-05-08T23:06:39.465Z" },
    "updatedAt": { "$date": "2025-05-08T23:06:39.465Z" }
  }
  // ... more transaction objects
]
```
Note the use of `userWallet` for the wallet address, `action` for transaction type, and `timestamp` as a Unix epoch integer. The `value_usd` is derived from `actionData.amount` and `actionData.assetPriceUSD`, assuming 6 decimal places for tokens like USDC.

## 🚀 Usage
### 1. 🏋️ Train the Model
Navigate to the project root directory and run the training script:
```bash
(venv) PS C:\Users\User\Downloads\Data Analysis> python src/train_model.py
```
This script will:
- 📂 Load `transactions.json`.
- 🔧 Engineer features for each wallet.
- 🏷️ Assign heuristic scores.
- 📈 Train the XGBoost Regressor.
- 📊 Evaluate the model's performance (MAE, R²).
- 💾 Save the trained model (`credit_score_model.pkl`), feature scaler (`feature_scaler.pkl`), output score scaler (`score_output_scaler.pkl`), and feature columns list (`feature_columns.pkl`) to the `src/models/` directory.

You should see output similar to:
```
Starting model training process...
Loaded 100000 transactions.
Engineering features for all wallets...
Engineered features for 3497 unique wallets.
Assigning heuristic scores...
Training on 2797 samples, testing on 700 samples.
Feature scaler trained and applied.
Training XGBoost Regressor...
XGBoost Regressor training complete.

Model Evaluation:
Mean Absolute Error (MAE): 3.67
R-squared (R²): 0.99

Model and scalers saved to src/models/
Training process finished.
```

### 2. 📊 Generate Credit Scores
Once the model is trained and saved, you can generate credit scores for all wallets:
```bash
(venv) PS C:\Users\User\Downloads\Data Analysis> python src/score_generator.py
```
This script will:
- 📂 Load `transactions.json` and engineer features.
- 📥 Load the pre-trained model and scalers from `src/models/`.
- 📊 Predict credit scores for all wallets.
- 📤 Save the results to `wallet_credit_scores.csv`.
- 📈 Generate and save a distribution plot of the credit scores as `score_distribution.png`.

You should see output similar to:
```
Starting credit score generation process...
Loaded 100000 transactions.
Engineering features...
Engineered features for 3497 unique wallets.
Scaling features...
Predicting credit scores...

Sample Wallet Scores:
... (sample scores) ...

Wallet scores saved to wallet_credit_scores.csv
Score distribution graph saved to score_distribution.png

Processing complete.
```

## 📈 Results
After running `score_generator.py`, you will find two new files in your project root directory:
- 📄 `wallet_credit_scores.csv`: A CSV file containing `wallet_address` and their corresponding `credit_score`.
- 📊 `score_distribution.png`: A histogram visualizing the distribution of the generated credit scores across all wallets.

## 📊 Analysis
A detailed analysis of the scored wallets will be provided in `analyze.md`. This will include:
- Score distribution graph across ranges (0-100, 100-200, etc.).
- Behavior of wallets in the lower score range.
- Behavior of wallets in the higher score range.

## 🤝 Contributing
Contributions are welcome! If you have suggestions for improving the feature engineering, heuristic scoring, model performance, or any other aspect of the project, feel free to open an issue or submit a pull request.

## 📜 License
This project is open-source and available under the MIT License.
