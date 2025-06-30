from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_predictions(df, label_col, prediction_col, label_name=""):
    """
    Evaluates classification results using standard metrics.

    Args:
        df (DataFrame): Pandas DataFrame with true and predicted labels.
        label_col (str): Column name for true labels (e.g., 'TT Manual').
        prediction_col (str): Column name for predicted values (e.g., 'is_technology_transfer_7').
        label_name (str): Optional label name for printout (e.g., 'TT' or 'CB').

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    y_true = df[label_col].astype(int)
    y_pred = df[prediction_col].astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()  # For easier export
    }

    print(f"Evaluation for {label_name or prediction_col}:")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"Accuracy:  {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall:    {metrics['recall']:.2f}")
    print(f"F1 Score:  {metrics['f1']:.2f}")

    return metrics
