from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    return report, cm


# Calculate Mean Squared Error (MSE)
def get_error(y_test, y_pred_test):
    mse_train = mean_squared_error(y_test, y_pred_test)
    print(f"Training MSE: {mse_train}")
    
    
def evaluate_model(model, x_test, y_test, y_pred_test):
    # Compute metrics
    test_loss, test_acc = model.evaluate(x_test, y_test)
    cm = confusion_matrix(y_test, y_pred_test)
    
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    precision = precision_score(y_test, y_pred_test, average='weighted')
    recall = recall_score(y_test, y_pred_test, average='weighted')
    
    # Ensure `class_names` is correctly defined for the classification report
    class_names = ['0', '1']  # Adjust if you have specific class names
    
    report = classification_report(y_test, y_pred_test, target_names=class_names)
    print(report)
    
    return test_loss, test_acc, cm, f1, precision, recall


def display_and_save_confution_matrix(cm, file_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(file_path)
    plt.show()