import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, confusion_matrix, \
    recall_score, f1_score
from cleandataset import clean_data


X = clean_data.drop('Diabetic', axis=1)
y = clean_data['Diabetic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=86)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_values = [3, 5, 7, 11, 13]
results = {'k': [], 'Accuracy': [], 'ROC AUC Score': [], 'Sensitivity': [], 'Specificity': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

for k in k_values:
    knnClassifier = KNeighborsClassifier(n_neighbors=k)
    knnClassifier.fit(X_train_scaled, y_train)
    y_pred = knnClassifier.predict(X_test_scaled)

    predected = knnClassifier.predict_proba(X_test_scaled)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, predected)
    ROC_AUC = auc(fpr, tpr)

    accuracy_knn = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f'\nConfusion Matrix for k={k}:')
    print(conf_matrix)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results['k'].append(k)
    results['Accuracy'].append(accuracy_knn)
    results['ROC AUC Score'].append(ROC_AUC)
    results['Sensitivity'].append(sensitivity)
    results['Specificity'].append(specificity)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1 Score'].append(f1)

result_table = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
print('\nResults for Different Values of k:')
print(result_table)
