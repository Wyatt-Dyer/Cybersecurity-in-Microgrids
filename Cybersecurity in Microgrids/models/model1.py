import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score

train_set = r'/Cybersecurity-in-Microgrids/Cybersecurity in Microgrids/data/raw/UNSW-NB15_c/UNSW_NB15_training-set.csv'
test_set = r'/Cybersecurity-in-Microgrids/Cybersecurity in Microgrids/data/raw/UNSW-NB15_c/UNSW_NB15_testing-set.csv'

df = pd.read_csv(train_set)
test_df = pd.read_csv(test_set)
df.head()

print(df.columns)

attack_labels = ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS']

generic_attacks = ['Generic']
exploits_attacks = ['Exploits']
fuzzers_attacks = ['Fuzzers']
dos_attacks = ['DoS']

def map_attack(attack):
    if attack in generic_attacks:
        attack_type = 1
    elif attack in exploits_attacks:
        attack_type = 2
    elif attack in fuzzers_attacks:
        attack_type = 3
    elif attack in dos_attacks:
        attack_type = 4
    else:
        attack_type = 0

    return attack_type

attack_map = df['attack_cat'].apply(map_attack)
df['attack_map'] = attack_map

test_attack_map = test_df['attack_cat'].apply(map_attack)
test_df['attack_map'] = test_attack_map

df.head()

features_to_encode = ['proto', 'service']
encoded = pd.get_dummies(df[features_to_encode])
test_encoded_base = pd.get_dummies(test_df[features_to_encode])

test_index = np.arange(len(test_df.index))
column_diffs = list(set(encoded.columns.values)-set(test_encoded_base.columns.values))

diff_df = pd.DataFrame(0, index=test_index, columns=column_diffs)

column_order = encoded.columns.to_list()

test_encoded_temp = test_encoded_base.join(diff_df)

test_final = test_encoded_temp[column_order].fillna(0)

numeric_features = [
       'dur', 'spkts', 'dpkts', 'sbytes','dbytes', 
       'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
       'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
       'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
       'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
       'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
       'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
       'ct_srv_dst', 'is_sm_ips_ports'
]

to_fit = encoded.join(df[numeric_features])
test_set = test_final.join(test_df[numeric_features])

binary_y = df['label']
multi_y = df['attack_map']

test_binary_y = test_df['label']
test_multi_y = test_df['attack_map']

binary_train_X, binary_val_X, binary_train_y, binary_val_y = train_test_split(to_fit, binary_y, test_size=0.6)
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(to_fit, multi_y, test_size = 0.6)

binary_model = RandomForestClassifier()
binary_model.fit(binary_train_X, binary_train_y)
binary_predictions = binary_model.predict(binary_val_X)

# calculate and display our base accuracty
base_rf_score = accuracy_score(binary_predictions,binary_val_y)
base_rf_score
models = [
    ('RandomForest', RandomForestClassifier()),
    ('LogisticRegression', LogisticRegression(max_iter=250)),
    ('KNeighbors', KNeighborsClassifier()),
    ('XGBoost', XGBClassifier()),
    ('MLP', MLPClassifier(max_iter=100))
]

# An empty list to capture the performance of each model
model_comps = []

# Walk through the models and populate our list
for model_name, model in models:
    # Train the model
    model.fit(binary_train_X, binary_train_y)
    
    # Save the model to a file
    joblib.dump(model, f'{model_name}_model.pkl')
    
    # Evaluate the model using cross-validation
    accuracies = cross_val_score(model, binary_train_X, binary_train_y, scoring='accuracy')
    
    # Append the performance results to the list
    for count, accuracy in enumerate(accuracies):
        model_comps.append((model_name, count, accuracy))

# Create a DataFrame to visualize model performance
result_df = pd.DataFrame(model_comps, columns=['model_name', 'count', 'accuracy'])
result_df.pivot(index='count', columns='model_name', values='accuracy').boxplot(rot=45)

def add_predictions(data_set,predictions,y):
    prediction_series = pd.Series(predictions, index=y.index)

    predicted_vs_actual = data_set.assign(predicted=prediction_series)
    original_data = predicted_vs_actual.assign(actual=y).dropna()
    conf_matrix = confusion_matrix(original_data['actual'], 
                                   original_data['predicted'])
    
    base_errors = original_data[original_data['actual'] != original_data['predicted']]
    
    non_zeros = base_errors.loc[:,(base_errors != 0).any(axis=0)]

    false_positives = non_zeros.loc[non_zeros.actual==0]
    false_negatives = non_zeros.loc[non_zeros.actual==1]

    prediction_data = {'data': original_data,
                       'confusion_matrix': conf_matrix,
                       'errors': base_errors,
                       'non_zeros': non_zeros,
                       'false_positives': false_positives,
                       'false_negatives': false_negatives}
    
    return prediction_data

# Load models
rf_model = joblib.load(r'/Cybersecurity-in-Microgrids/Cybersecurity in Microgrids/data/processed/RandomForest_model.pkl')
lr_model = joblib.load(r'/Cybersecurity-in-Microgrids/Cybersecurity in Microgrids/data/processed/LogisticRegression_model.pkl')
knn_model = joblib.load(r'/Cybersecurity-in-Microgrids/Cybersecurity in Microgrids/data/processed/KNeighbors_model.pkl')
xgb_model = joblib.load(r'/Cybersecurity-in-Microgrids/Cybersecurity in Microgrids/data/processed/XGBoost_model.pkl')
mlp_model = joblib.load(r'/Cybersecurity-in-Microgrids/Cybersecurity in Microgrids/data/processed/MLP_model.pkl')

# Function to evaluate the model and store results in a dictionary
def evaluate_model(model, X_train, y_train, X_val, y_val, model_name, results_dict):
    predictions = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, average='binary')
    recall = recall_score(y_val, predictions, average='binary')
    f1 = f1_score(y_val, predictions, average='binary')
    roc_auc = roc_auc_score(y_val, probs) if probs is not None else np.nan  # ROC-AUC requires probability scores
    logloss = log_loss(y_val, probs) if probs is not None else np.nan  # Log loss requires probability scores

    # Store the results in the dictionary
    results_dict[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Log Loss': logloss
    }

    # Confusion Matrix and Classification Report
    conf_matrix = confusion_matrix(y_val, predictions)
    class_report = classification_report(y_val, predictions, target_names=['Normal', 'DDos Attack'])

    # Print results with alignment
    print(f"{'Model:'.ljust(20)} {model_name}")
    print(f"{'Accuracy:'.ljust(20)} {accuracy:.4f}")
    print(f"{'Precision:'.ljust(20)} {precision:.4f}")
    print(f"{'Recall:'.ljust(20)} {recall:.4f}")
    print(f"{'F1-Score:'.ljust(20)} {f1:.4f}")
    print(f"{'ROC-AUC:'.ljust(20)} {roc_auc:.4f}")
    print(f"{'Log Loss:'.ljust(20)} {logloss:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    
    # Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=['Predicted Normal', 'Predicted Attack'],
                yticklabels=['Actual Normal', 'Actual Attack'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

# Dictionary to store the results for each model
results = {}

# Evaluate each model
evaluate_model(rf_model, binary_train_X, binary_train_y, binary_val_X, binary_val_y, 'Random Forest', results)
evaluate_model(lr_model, binary_train_X, binary_train_y, binary_val_X, binary_val_y, 'Logistic Regression', results)
evaluate_model(knn_model, binary_train_X, binary_train_y, binary_val_X, binary_val_y, 'K-Nearest Neighbors', results)
evaluate_model(xgb_model, binary_train_X, binary_train_y, binary_val_X, binary_val_y, 'XGBoost Classifier', results)
evaluate_model(mlp_model, binary_train_X, binary_train_y, binary_val_X, binary_val_y, 'Neural Network', results)

# Convert the results dictionary to a pandas DataFrame
results_df = pd.DataFrame(results).T  # Transpose to have models as rows and metrics as columns

# Save the results to a CSV file
results_df.to_csv('/Cybersecurity-in-Microgrids/Cybersecurity in Microgrids/data/processed/UNSW_results.csv')

# Display the results DataFrame
print(results_df)