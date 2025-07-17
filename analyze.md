# Crypto Wallet Credit Scoring Project Analysis

## 1. Introduction and Objective

This project aimed to develop a machine learning model capable of assigning a "credit score" to cryptocurrency wallets based on their on-chain transaction history. The primary objective was to transform raw transaction data into meaningful features and then use these features to predict a heuristic score, ultimately providing a quantitative measure of a wallet's engagement and perceived "trustworthiness" within the decentralized lending ecosystem.

## 2. Data Source

The project utilized a dataset of `100,000` blockchain transactions stored in `data/transactions.json`. This raw transaction data served as the foundation for extracting behavioral patterns of individual wallets.

## 3. Methodology

The credit scoring system was built through a multi-stage process:

### 3.1. Feature Engineering

Raw transaction data was aggregated and transformed into a comprehensive set of features for each unique wallet. Key engineered features included:
* `total_transactions`: Total number of transactions.
* `unique_transaction_types`: Diversity of actions performed (e.g., deposit, borrow, repay).
* `time_since_first_tx_days`: Longevity of wallet activity.
* `avg_time_between_tx_days`: Frequency of transactions.
* `total_deposited_value_usd`, `total_borrowed_value_usd`, `total_repaid_value_usd`, `total_redeemed_value_usd`: Cumulative financial activity.
* `net_deposit_value_usd`, `net_borrow_value_usd`: Net financial flows.
* `num_liquidations_as_borrower`, `num_liquidations_as_liquidator`: Indicators of risk or active participation in liquidations.
* `borrow_to_deposit_ratio`, `repay_to_borrow_ratio`, `deposit_reversal_ratio`: Ratios indicating specific behavioral patterns (e.g., repayment discipline, deposit retention).

A total of `3497` unique wallets had sufficient data for feature engineering.

### 3.2. Heuristic Labeling

Since real-world "credit scores" for crypto wallets are not readily available as ground truth, a **heuristic scoring mechanism** was implemented. This rule-based system assigned a score from 0 to 1000 to each wallet based on a predefined set of weighted criteria derived from the engineered features. For example:
* Liquidations as a borrower significantly decreased the score.
* High repayment ratios and net positive deposits increased the score.
* Activity longevity and high transaction count had a positive impact.
* High borrow-to-deposit ratios combined with low repayment ratios negatively impacted the score.

This heuristic score served as the **target variable** for the machine learning model.

### 3.3. Model Training

An **XGBoost Regressor** was chosen as the machine learning model.
* **Data Splitting:** The engineered features and heuristic scores were split into training (80%) and testing (20%) sets.
* **Feature Scaling:** `MinMaxScaler` was applied to the features to normalize their range, which can improve model performance and stability for some algorithms.
* **Model Configuration:** The XGBoost Regressor was configured with `n_estimators=1000`, `learning_rate=0.05`, `max_depth=7`, and other standard parameters.
* **Early Stopping:** To prevent overfitting and optimize training time, `early_stopping_rounds=50` was used. This feature, correctly implemented in the `XGBRegressor` constructor for modern XGBoost versions, stopped training if the performance on the validation set (`eval_set`) did not improve for 50 consecutive rounds.

## 4. Model Performance

After training, the model's performance on the unseen test set was evaluated:

* **Mean Absolute Error (MAE): `3.67`**
    This indicates that, on average, the model's predicted credit scores were only 3.67 points away from the heuristic scores.
* **R-squared (R²): `0.99`**
    An R-squared value of 0.99 is exceptionally high, meaning that 99% of the variance in the heuristic scores can be explained by the features used by the model. This suggests the XGBoost model has learned to accurately replicate the logic of the heuristic scoring function.

The high performance metrics demonstrate that the XGBoost Regressor effectively learned the complex non-linear relationships embedded in the heuristic scoring rules.

## 5. Score Generation and Analysis

Following successful training and evaluation, the pipeline was used to generate credit scores for all `3497` unique wallets:

* The trained `credit_score_model.pkl`, `feature_scaler.pkl`, and `feature_columns.pkl` were loaded.
* New (or all existing) wallet features were scaled using the trained scaler.
* Predictions were made using the loaded XGBoost model.
* The raw predicted scores were then scaled to a range of 0-1000 using `score_output_scaler.pkl`.

### Sample Wallet Scores:

| wallet_address                         | credit_score |
| :------------------------------------- | :----------- |
| 0x00000000001accfa9cef68cf5371a23025b6d4b6 | 599.82       |
| 0x000000000051d07a4fb3bd10121a343d85818da6 | 600.03       |
| 0x000000000096026fb41fc39f9875d164bd82e2dc | 600.01       |
| 0x0000000000e189dd664b9ab08a33c4839953852c | 530.06       |
| 0x0000000002032370b971dabd36d72f3e5a7bf1ee | 500.02       |

The full list of scores is saved in `wallet_credit_scores.csv`. A visual representation of the score distribution is available in `score_distribution.png`. The sample scores show a range of values, demonstrating that the model is indeed differentiating between wallets based on their activity.

## 6. Limitations and Future Work

### 6.1. Heuristic Dependence

The primary limitation of this project is its reliance on a heuristic scoring system for labeling. While the model accurately reproduces these heuristics, the ultimate "creditworthiness" is still defined by these hand-crafted rules, which may not fully capture real-world financial risk.

### 6.2. Data Scope

The analysis is limited to the provided transaction data. A more comprehensive system would integrate data from multiple chains, protocols, and potentially off-chain sources (if privacy-preserving).

### 6.3. Feature Complexity

While a good set of features was engineered, more advanced time-series analysis, network graph analysis (interactions between wallets), or incorporating liquidity pool data could yield richer insights.

### 6.4. Future Enhancements

* **Real Ground Truth:** The ideal next step would be to acquire (or simulate) actual "default" or "repayment" labels for wallets to train a truly predictive credit risk model, moving beyond a heuristic replication.
* **Explainability:** Incorporating model explainability techniques (e.g., SHAP values) could provide insights into which specific wallet behaviors contribute most to their credit score.
* **Dynamic Scoring:** Implementing a system for real-time or near real-time score updates as new transactions occur.
* **Risk Categorization:** Instead of a single score, classifying wallets into risk categories (e.g., low, medium, high risk).