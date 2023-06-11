from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd



df_final_training_set = pd.read_csv('Training_set.csv')
df_final_test_set = pd.read_csv('Test_set.csv')

X_train = df_final_training_set.drop(columns=['is_depressive', 'file_name', 'id'])
y_train = df_final_training_set['is_depressive']

X_test = df_final_test_set.drop(columns=['is_depressive', 'file_name', 'id'])
y_test = df_final_test_set['is_depressive']

param_grid_tree = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}


# Decision Tree Classifier
print("\nDecision Tree:")
clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_tree, cv=5, scoring='accuracy')
clf.fit(X_train, y_train)

selector = SelectFromModel(clf.best_estimator_, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print("Number of features selected (Decision Tree):", X_train_selected.shape[1])
print("Best parameters (Decision Tree):", clf.best_params_)

# Perform cross-validation on Decision Tree classifier
scores = cross_val_score(clf.best_estimator_, X_train_selected, y_train, cv=5, scoring='accuracy')
print("Cross-validation scores (Decision Tree):", scores)
print("Mean cross-validation score (Decision Tree):", np.mean(scores))

clf_best = clf.best_estimator_
clf_best.fit(X_train_selected, y_train)
y_pred_selected = clf_best.predict(X_test_selected)

print("Test set accuracy (selected features):", accuracy_score(y_test, y_pred_selected))
print(classification_report(y_test, y_pred_selected))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_selected))