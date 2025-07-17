import json
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
TRANSACTION_FILE = 'data/transactions.json' # Input data for scoring
MODELS_DIR = 'src/models'
PRETRAINED_MODEL_PATH = os.path.join(MODELS_DIR, 'credit_score_model.pkl')
FEATURE_SCALER_PATH = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
SCORE_OUTPUT_SCALER_PATH = os.path.join(MODELS_DIR, 'score_output_scaler.pkl')
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, 'feature_columns.pkl') # Path to saved feature columns
OUTPUT_CSV_FILE = 'wallet_credit_scores.csv'
SCORE_DISTRIBUTION_PLOT = 'score_distribution.png'

# --- Feature Engineering Function (Identical to train_model for consistency) ---
def engineer_features(transactions_data):
    wallet_features = {}

    for tx in transactions_data:
        # **UPDATE 1: Use 'userWallet' as the wallet address key**
        wallet = tx.get('userWallet')
        if not wallet: # Skip if userWallet is missing or None
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
        
        # Safely parse timestamp
        try:
            # **UPDATE 2: Specify unit='s' for Unix epoch timestamps**
            timestamp_val = tx.get('timestamp')
            if timestamp_val is not None: # Ensure timestamp exists before trying to convert
                 wallet_features[wallet]['timestamps'].append(pd.to_datetime(timestamp_val, unit='s'))
            else:
                pass # Optionally log if a transaction is missing a timestamp
        except (ValueError, TypeError) as e:
            # Handle invalid timestamps, e.g., skip or log
            # print(f"DEBUG: Invalid timestamp format for transaction (txHash: {tx.get('txHash', 'N/A')}): {e}")
            continue # Skip this transaction's timestamp if it's unparseable

        # Ensure transaction_type exists before adding to set
        tx_type = tx.get('action') # **UPDATE 3: Use 'action' for transaction_type**
        if tx_type:
            wallet_features[wallet]['unique_transaction_types'].add(tx_type)

        # **UPDATE 4: Use 'actionData.amount' and 'actionData.assetPriceUSD' to calculate value_usd**
        action_data = tx.get('actionData', {})
        amount_str = action_data.get('amount')
        asset_price_usd_str = action_data.get('assetPriceUSD')
        
        value_usd = 0.0
        try:
            if amount_str and asset_price_usd_str:
                # Assuming amount is in wei-like format (e.g., 2000000000 for 2 USDC if USDC has 6 decimals)
                # ADJUST THIS DIVISOR based on the actual decimals of the assets in your JSON data.
                # For USDC, 10**6 is common. For ETH, it's 10**18.
                amount_normalized = float(amount_str) / (10**6) 
                value_usd = amount_normalized * float(asset_price_usd_str)
            elif tx.get('value_usd') is not None: # Fallback if a root value_usd exists for other tx types
                 value_usd = float(tx['value_usd'])
        except (ValueError, TypeError):
            value_usd = 0.0 # Default to 0 if conversion fails


        # **UPDATE 5: Use 'action' for transaction_type in conditions**
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
            # **UPDATE 6: Check 'actionData' for borrower/liquidator**
            if action_data.get('borrower') == wallet:
                wallet_features[wallet]['num_liquidations_as_borrower'] += 1
            elif action_data.get('liquidator') == wallet:
                wallet_features[wallet]['num_liquidations_as_liquidator'] += 1

    final_features = []
    for wallet, data in wallet_features.items():
        if not data['timestamps']:
            continue # Skip wallets with no valid timestamps

        first_tx_time = min(data['timestamps'])
        last_tx_time = max(data['timestamps'])
        
        # Sort timestamps for accurate time difference calculations
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
    
    # Handle the case where final_features might be empty
    if not final_features:
        print("Warning: No valid wallets found with sufficient data to engineer features. Returning empty DataFrame.")
        # Create an empty DataFrame with the expected columns, even if no data
        return pd.DataFrame(columns=[
            'wallet_address', 'total_transactions', 'unique_transaction_types',
            'time_since_first_tx_days', 'avg_time_between_tx_days', 'tx_frequency_per_day',
            'total_deposited_value_usd', 'total_borrowed_value_usd', 'total_repaid_value_usd',
            'total_redeemed_value_usd', 'net_deposit_value_usd', 'net_borrow_value_usd',
            'num_liquidations_as_borrower', 'num_liquidations_as_liquidator',
            'borrow_to_deposit_ratio', 'repay_to_borrow_ratio', 'deposit_reversal_ratio'
        ]).set_index('wallet_address')

    return pd.DataFrame(final_features).set_index('wallet_address')


# --- Main Script ---
if __name__ == "__main__":
    print("Starting credit score generation process...")
    print("Loading transaction data...")
    try:
        with open(TRANSACTION_FILE, 'r') as f:
            transactions = json.load(f)
    except FileNotFoundError:
        print(f"Error: {TRANSACTION_FILE} not found. Please ensure it's in the 'data/' directory.")
        print("Run 'python src/train_model.py' first to ensure data is present and models are trained.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {TRANSACTION_FILE}. Check file integrity.")
        exit(1)
        
    print(f"Loaded {len(transactions)} transactions.")

    print("Engineering features...")
    features_df = engineer_features(transactions)
    print(f"Engineered features for {len(features_df)} unique wallets.")

    # Drop any rows with NaN or Inf values that might have resulted from edge cases in feature engineering
    features_df = features_df.replace([float('inf'), -float('inf')], 0) 
    features_df = features_df.fillna(0) 

    # Load pre-trained model and scalers
    try:
        model = joblib.load(PRETRAINED_MODEL_PATH)
        feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        score_output_scaler = joblib.load(SCORE_OUTPUT_SCALER_PATH)
        model_feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    except FileNotFoundError:
        print(f"Error: One or more model files not found in {MODELS_DIR}/.")
        print("Please run 'python src/train_model.py' first to train and save the model and scalers.")
        exit(1)
    except Exception as e:
        print(f"Error loading model components: {e}")
        exit(1)

    # Handle the case where no features could be engineered (e.g., empty or malformed input JSON)
    if features_df.empty:
        print("No features could be engineered from the provided transactions. No scores to generate.")
        # Create an empty scores DataFrame and plot to avoid further errors
        wallet_scores_df = pd.DataFrame(columns=['wallet_address', 'credit_score'])
        wallet_scores_df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Empty wallet scores saved to {OUTPUT_CSV_FILE}")
        
        plt.figure(figsize=(10, 6))
        plt.title('Distribution of Wallet Credit Scores (0-1000) - No Data', fontsize=16)
        plt.xlabel('Credit Score Range', fontsize=12)
        plt.ylabel('Number of Wallets', fontsize=12)
        plt.text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=20, color='gray')
        plt.savefig(SCORE_DISTRIBUTION_PLOT)
        print(f"Empty score distribution graph saved to {SCORE_DISTRIBUTION_PLOT}")
        exit(0) # Exit successfully as no data was the expected scenario

    # Reindex the features_df to match the order of columns used during training
    current_features_columns = features_df.columns.tolist()
    missing_in_current = [col for col in model_feature_columns if col not in current_features_columns]
    extra_in_current = [col for col in current_features_columns if col not in model_feature_columns]

    if missing_in_current:
        print(f"Warning: The following features used during training are missing in the current data and will be filled with zeros: {missing_in_current}")
        for col in missing_in_current:
            features_df[col] = 0.0

    if extra_in_current:
        print(f"Warning: The following new features are in the current data but were not in the training data and will be ignored: {extra_in_current}")
        features_df = features_df.drop(columns=extra_in_current)

    # Ensure the order of columns matches the training data
    features_df = features_df[model_feature_columns]

    print("Scaling features...")
    scaled_features = feature_scaler.transform(features_df)
    # Convert back to DataFrame to preserve wallet_addresses as index for later
    scaled_features_df = pd.DataFrame(scaled_features, columns=model_feature_columns, index=features_df.index)

    print("Predicting credit scores...")
    raw_scores = model.predict(scaled_features_df)

    # Transform raw scores to the 0-1000 range using the pre-trained output scaler
    # Since heuristic labels are 0-1000, model predicts in that range. We just clip.
    final_scores = raw_scores.clip(0, 1000)


    wallet_scores_df = pd.DataFrame({
        'wallet_address': features_df.index,
        'credit_score': final_scores
    })

    print("\nSample Wallet Scores:")
    print(wallet_scores_df.head())

    # Save scores to CSV
    wallet_scores_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\nWallet scores saved to {OUTPUT_CSV_FILE}")

    # Generate and save score distribution graph
    plt.figure(figsize=(10, 6))
    sns.histplot(wallet_scores_df['credit_score'], bins=20, kde=True, palette='viridis')
    plt.title('Distribution of Wallet Credit Scores (0-1000)', fontsize=16)
    plt.xlabel('Credit Score Range', fontsize=12)
    plt.ylabel('Number of Wallets', fontsize=12)
    plt.xticks(range(0, 1001, 100))
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(SCORE_DISTRIBUTION_PLOT)
    print(f"Score distribution graph saved to {SCORE_DISTRIBUTION_PLOT}")
    print("\nProcessing complete.")