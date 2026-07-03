import pickle
import pandas as pd
from .preprocessing import clean_data, encode_features
from .config import MODEL_PATH, TARGET_CLASSES


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


def align_columns(df, model):
    if hasattr(model, 'feature_names_in_'):
        cols = [c for c in model.feature_names_in_ if c in df.columns]
        return df[cols]
    return df


def predict_single(model, input_dict):
    """
    Predict rainfall type for a single farmer submission.
    Returns dict with prediction (int), label (NORAIN/SMALLRAIN/etc), probabilities.
    """
    df = pd.DataFrame([input_dict])
    df = clean_data(df)
    df = encode_features(df)
    df = align_columns(df, model)

    prediction = int(model.predict(df)[0])
    probabilities = model.predict_proba(df)[0]

    return {
        'prediction': prediction,
        'label': TARGET_CLASSES[prediction],
        'probabilities': dict(zip(TARGET_CLASSES, probabilities))
    }


def predict_batch(model, df):
    """
    Predict rainfall type for a batch of submissions.
    Returns df with Prediction and Label columns added.
    """
    df = df.copy()
    df = clean_data(df)
    df = encode_features(df)
    df = align_columns(df, model)

    predictions = model.predict(df)
    df['Prediction'] = predictions
    df['Label'] = df['Prediction'].map(lambda x: TARGET_CLASSES[x])

    return df


if __name__ == '__main__':
    pass
