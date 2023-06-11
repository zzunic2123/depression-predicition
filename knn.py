from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_final_training_set = pd.read_csv('Training_set.csv')
df_final_test_set = pd.read_csv('Test_set.csv')

# split data into X (features) and y (target)
X_train = df_final_training_set.drop(columns=['is_depressive', 'file_name', 'id'])
y_train = df_final_training_set['is_depressive']

X_test = df_final_test_set.drop(columns=['is_depressive', 'file_name', 'id'])
y_test = df_final_test_set['is_depressive']

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}


# Feature selection for KNN using RandomForestClassifier
print("\nFeature selection for KNN:")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

selector_knn = SelectFromModel(rf, threshold=0.01)
selector_knn.fit(X_train, y_train)

X_train_selected_knn = selector_knn.transform(X_train)
X_test_selected_knn = selector_knn.transform(X_test)

print("Number of features selected (KNN):", X_train_selected_knn.shape[1])

# GridSearchCV for KNN with selected features
knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
knn.fit(X_train_selected_knn, y_train)

print("Best parameters (K-Nearest Neighbors with selected features):", knn.best_params_)

# Perform cross-validation on K-Nearest Neighbors classifier with selected features
scores = cross_val_score(knn.best_estimator_, X_train_selected_knn, y_train, cv=5, scoring='accuracy')
print("Cross-validation scores (K-Nearest Neighbors with selected features):", scores)
print("Mean cross-validation score (K-Nearest Neighbors with selected features):", np.mean(scores))

knn_best = knn.best_estimator_
y_pred_selected_knn = knn_best.predict(X_test_selected_knn)

print("Test set accuracy (KNN with selected features):", accuracy_score(y_test, y_pred_selected_knn))
print(classification_report(y_test, y_pred_selected_knn))
print("Confusion Matrix:")

cm = confusion_matrix(y_test, y_pred_selected_knn)
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.savefig('confusion_matrix_knn_selected.png')