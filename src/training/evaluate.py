from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Computes standard evaluation metrics.
    If y_prob is provided, computes ROC-AUC as well.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='binary'),
        "recall": recall_score(y_true, y_pred, average='binary'),
        "f1": f1_score(y_true, y_pred, average='binary')
    }
    
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        
    return metrics

def print_evaluation_report(y_true, y_pred):
    """Prints a detailed classification report and confusion matrix."""
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Fake (0)", "Real (1)"]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
