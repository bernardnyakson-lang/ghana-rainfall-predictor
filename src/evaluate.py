import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from sklearn.metrics import (  # type: ignore[import]
    classification_report, confusion_matrix, f1_score
)

# support both direct script execution and package imports
try:
    from .preprocessing import load_data, prepare_features
    from .config import MODEL_PATH, TARGET_CLASSES
except ImportError:
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from preprocessing import load_data, prepare_features
    from config import MODEL_PATH, TARGET_CLASSES


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    Print classification report with all four rainfall classes.
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"F1 Macro Score: {f1:.4f}")
    return y_pred


def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix for all four classes."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=TARGET_CLASSES,
                yticklabels=TARGET_CLASSES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = load_data()
    _, X_test, _, y_test = prepare_features(df)
    model = load_model()
    y_pred = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plt.show()
