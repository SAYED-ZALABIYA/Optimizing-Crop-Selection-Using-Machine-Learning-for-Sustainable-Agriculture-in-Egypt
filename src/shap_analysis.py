import matplotlib.pyplot as plt
import warnings
!pip install shap
import shap

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 5))
sns.countplot(df['Crop'])
plt.title('Distribution of Crops')
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Predicted vs Actual Data Plot
y_pred = best_model.predict(x_test)
plt.scatter(y_pred, y_test, color='blue', label='Data Points')
plt.plot([min(y_pred), max(y_pred)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Predictions')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Predicted vs Actual Data')
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = df2.columns
plt.figure()
plt.title("Feature Importances")
plt.bar(range(x_train.shape[1]), importances[indices], align="center")
plt.xticks(range(x_train.shape[1]), features[indices], rotation=90)
plt.show()

# Model Interpretation with SHAP
#explainer = shap.TreeExplainer(best_model)
#shap_values = explainer.shap_values(x_test)
#shap.summary_plot(shap_values, x_test, feature_names=df2.columns)
explainer = shap.Explainer(best_model)
shap_values = explainer(x_test)
# Check shapes
print(f"x_test shape: {x_test.shape}")
print(f"shap_values shape: {shap_values.shape}")

# Ensure correct feature names
feature_names = df2.columns[:x_test.shape[0]]

# Plotting SHAP summary
#shap.summary_plot(shap_values, x_test, feature_names=feature_names)
class_index = 0  
shap_values_for_class = shap_values[:, :, class_index]  
shap.summary_plot(shap_values_for_class, x_test, feature_names=feature_names)