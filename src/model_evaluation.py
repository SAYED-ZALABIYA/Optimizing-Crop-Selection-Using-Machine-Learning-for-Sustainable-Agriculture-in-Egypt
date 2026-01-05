from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Model evaluation on the test data
y_pred_test = best_model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Score (Accuracy): {test_accuracy}")

print("Classification Report:")
print(classification_report(y_test, y_pred_test))