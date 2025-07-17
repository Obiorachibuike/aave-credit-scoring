import json
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor # Make sure this is imported
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import sys
# from xgboost.callback import EarlyStopping # This line should now be commented out or removed
import xgboost as xgb

print(f"--- Debug Info ---")
print(f"Python executable: {sys.executable}")
print(f"XGBoost version (at runtime): {xgb.__version__}")
print(f"--- End Debug Info ---")


# --- Configuration ---
TRANSACTION_FILE = 'data/transactions.json'
MODELS_DIR = 'src/models'

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Feature Engineering Function ---
def engineer_features(transactions_data):
    wallet_features = {}

    for tx in transactions_data:
        wallet = tx.get('userWallet')
        if not wallet:
            continue

        if wallet not in wallet_features:
            wallet_features[wallet] = {
                'transactions': [],
                'total_deposited_value_usd': 0.0,
                'total_borrowed_value_usd': 0.0,
                'total_repaid_value_usd': 0.0,
                'total_redeemed_value_usd': 0.0,
                'num_liquidations_as_borrower': 0,
                'num_liquidations_as_liquidator': 0,
                'unique_transaction_types': set(),
                'timestamps': []
            }

        wallet_features[wallet]['transactions'].append(tx)
        
        try:
            timestamp_val = tx.get('timestamp')
            if timestamp_val is not None:
                wallet_features[wallet]['timestamps'].append(pd.to_datetime(timestamp_val, unit='s'))
        except (ValueError, TypeError):
            continue

        tx_type = tx.get('action')
        if tx_type:
            wallet_features[wallet]['unique_transaction_types'].add(tx_type)

        action_data = tx.get('actionData', {})
        amount_str = action_data.get('amount')
        asset_price_usd_str = action_data.get('assetPriceUSD')
        
        value_usd = 0.0
        try:
            if amount_str and asset_price_usd_str:
                amount_normalized = float(amount_str) / (10**6)
                value_usd = amount_normalized * float(asset_price_usd_str)
            elif tx.get('value_usd') is not None:
                value_usd = float(tx['value_usd'])
        except (ValueError, TypeError):
            value_usd = 0.0

        action = tx.get('action')
        if action == 'deposit':
            wallet_features[wallet]['total_deposited_value_usd'] += value_usd
        elif action == 'borrow':
            wallet_features[wallet]['total_borrowed_value_usd'] += value_usd
        elif action == 'repay':
            wallet_features[wallet]['total_repaid_value_usd'] += value_usd
        elif action == 'redeemunderlying':
            wallet_features[wallet]['total_redeemed_value_usd'] += value_usd
        elif action == 'liquidationCall':
            if action_data.get('borrower') == wallet:
                wallet_features[wallet]['num_liquidations_as_borrower'] += 1
            elif action_data.get('liquidator') == wallet:
                wallet_features[wallet]['num_liquidations_as_liquidator'] += 1

    final_features = []
    for wallet, data in wallet_features.items():
        if not data['timestamps']:
            continue

        first_tx_time = min(data['timestamps'])
        last_tx_time = max(data['timestamps'])

        data['timestamps'].sort()
        time_diffs_seconds = [(data['timestamps'][i] - data['timestamps'][i-1]).total_seconds()
                              for i in range(1, len(data['timestamps']))]
        avg_time_between_tx_days = (sum(time_diffs_seconds) / len(time_diffs_seconds) / (3600 * 24)) if time_diffs_seconds else 0.0

        time_since_first_tx_days = (last_tx_time - first_tx_time).total_seconds() / (3600 * 24)
        total_transactions = len(data['transactions'])

        tx_frequency_per_day = total_transactions / (time_since_first_tx_days + 1e-6)

        borrow_to_deposit_ratio = data['total_borrowed_value_usd'] / (data['total_deposited_value_usd'] + 1e-6)
        repay_to_borrow_ratio = data['total_repaid_value_usd'] / (data['total_borrowed_value_usd'] + 1e-6)
        deposit_reversal_ratio = data['total_redeemed_value_usd'] / (data['total_deposited_value_usd'] + 1e-6)

        borrow_to_deposit_ratio = min(borrow_to_deposit_ratio, 1000.0)
        repay_to_borrow_ratio = min(repay_to_borrow_ratio, 1000.0)
        deposit_reversal_ratio = min(deposit_reversal_ratio, 1000.0)

        final_features.append({
            'wallet_address': wallet,
            'total_transactions': float(total_transactions),
            'unique_transaction_types': float(len(data['unique_transaction_types'])),
            'time_since_first_tx_days': float(time_since_first_tx_days),
            'avg_time_between_tx_days': float(avg_time_between_tx_days),
            'tx_frequency_per_day': float(tx_frequency_per_day),
            'total_deposited_value_usd': data['total_deposited_value_usd'],
            'total_borrowed_value_usd': data['total_borrowed_value_usd'],
            'total_repaid_value_usd': data['total_repaid_value_usd'],
            'total_redeemed_value_usd': data['total_redeemed_value_usd'],
            'net_deposit_value_usd': data['total_deposited_value_usd'] - data['total_redeemed_value_usd'],
            'net_borrow_value_usd': data['total_borrowed_value_usd'] - data['total_repaid_value_usd'],
            'num_liquidations_as_borrower': float(data['num_liquidations_as_borrower']),
            'num_liquidations_as_liquidator': float(data['num_liquidations_as_liquidator']),
            'borrow_to_deposit_ratio': borrow_to_deposit_ratio,
            'repay_to_borrow_ratio': repay_to_borrow_ratio,
            'deposit_reversal_ratio': deposit_reversal_ratio,
        })

    if not final_features:
        print("Warning: No valid wallets found with sufficient data to engineer features.")
        return pd.DataFrame(columns=[
            'wallet_address', 'total_transactions', 'unique_transaction_types',
            'time_since_first_tx_days', 'avg_time_between_tx_days', 'tx_frequency_per_day',
            'total_deposited_value_usd', 'total_borrowed_value_usd', 'total_repaid_value_usd',
            'total_redeemed_value_usd', 'net_deposit_value_usd', 'net_borrow_value_usd',
            'num_liquidations_as_borrower', 'num_liquidations_as_liquidator',
            'borrow_to_deposit_ratio', 'repay_to_borrow_ratio', 'deposit_reversal_ratio'
        ]).set_index('wallet_address')

    return pd.DataFrame(final_features).set_index('wallet_address')


# --- Heuristic Labeling Function ---
def assign_heuristic_score(df):
    scores = []
    for index, row in df.iterrows():
        score = 500

        if row['num_liquidations_as_borrower'] > 0:
            score -= (row['num_liquidations_as_borrower'] * 150)
            if row['num_liquidations_as_borrower'] >= 2:
                score -= 100

        if row['total_borrowed_value_usd'] > 0:
            if row['repay_to_borrow_ratio'] >= 1.0:
                score += 150
            elif row['repay_to_borrow_ratio'] > 0.75:
                score += 100
            elif row['repay_to_borrow_ratio'] < 0.25:
                score -= 100
        elif row['total_borrowed_value_usd'] == 0 and row['total_deposited_value_usd'] > 0:
            score += 50

        if row['net_deposit_value_usd'] > 0 and row['total_deposited_value_usd'] > 1000:
            score += 50
        if row['time_since_first_tx_days'] > 30:
            score += 30
        if row['total_transactions'] > 50:
            score += 20

        if row['borrow_to_deposit_ratio'] > 1.5 and row['repay_to_borrow_ratio'] < 0.8:
            score -= 75
        if row['deposit_reversal_ratio'] > 0.8 and row['total_deposited_value_usd'] > 100:
            score -= 50

        if row['tx_frequency_per_day'] > 10 and row['total_transactions'] > 100 and row['total_deposited_value_usd'] < 100:
            score -= 100

        score = max(0, min(1000, score))
        scores.append(score)
    return pd.Series(scores, index=df.index)


# --- Main Training Script ---
if __name__ == "__main__":
    print("Starting model training process...")
    print("Loading transaction data...")
    try:
        with open(TRANSACTION_FILE, 'r') as f:
            transactions = json.load(f)
    except FileNotFoundError:
        print(f"Error: {TRANSACTION_FILE} not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {TRANSACTION_FILE}.")
        exit(1)

    print(f"Loaded {len(transactions)} transactions.")
    print("Engineering features for all wallets...")
    features_df = engineer_features(transactions)
    print(f"Engineered features for {len(features_df)} unique wallets.")

    features_df = features_df.replace([float('inf'), -float('inf')], 0).fillna(0)

    if 'heuristic_score' in features_df.columns:
        model_features = features_df.columns.drop('heuristic_score').tolist()
    else:
        model_features = features_df.columns.tolist()

    if features_df.empty or not model_features:
        print("Error: No features could be engineered.")
        exit(1)

    print("Assigning heuristic scores...")
    features_df['heuristic_score'] = assign_heuristic_score(features_df)

    X = features_df[model_features]
    y = features_df['heuristic_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    print("Feature scaler trained and applied.")

    print("Training XGBoost Regressor...")
    # UPDATED: early_stopping_rounds and eval_metric moved to constructor
    model = XGBRegressor(n_estimators=1000,
                         learning_rate=0.05,
                         max_depth=7,
                         subsample=0.7,
                         colsample_bytree=0.7,
                         random_state=42,
                         n_jobs=-1,
                         early_stopping_rounds=50, # Moved to constructor
                         eval_metric='rmse'        # Moved to constructor, specifying the metric
                        )

    model.fit(X_train_scaled, y_train,
              eval_set=[(X_test_scaled, y_test)],
              verbose=False) # Removed 'callbacks' argument

    print("XGBoost Regressor training complete.")

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nModel Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    score_output_scaler = MinMaxScaler(feature_range=(0, 1000))
    score_output_scaler.fit([[0], [1000]])

    joblib.dump(model, os.path.join(MODELS_DIR, 'credit_score_model.pkl'))
    joblib.dump(feature_scaler, os.path.join(MODELS_DIR, 'feature_scaler.pkl'))
    joblib.dump(score_output_scaler, os.path.join(MODELS_DIR, 'score_output_scaler.pkl'))
    joblib.dump(model_features, os.path.join(MODELS_DIR, 'feature_columns.pkl'))

    print(f"\nModel and scalers saved to {MODELS_DIR}/")
    print("Training process finished.")