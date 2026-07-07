import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# import train_test_split where used to avoid static resolution issues in some editors
from src.config import (
    DATA_PATH, TARGET_COL, DROP_COLS,
    CATEGORICAL_COLS, TARGET_CLASSES, TEST_SIZE, RANDOM_STATE, NUMERICAL_COLS
)

def load_data(path=DATA_PATH):
    """Load the rainfall dataset from CSV."""
    return pd.read_csv(path)

def standardise_community(name):
    """Clean up inconsistent community name spellings/casing."""
    #if pd.isna(name):
       # return 'Unknown'
    name = ' '.join(name.strip().lower().split())
    mapping = {
        'akwaduuso': 'Akwaduuso',
        'foso odumasi': 'Foso Odumasi',
        'odumasi adansi': 'Odumasi',
        'assin foso odumasi': 'Foso Odumasi',
        'odumasi': 'Odumasi',
        'asamama': 'Asamama',
        'assin nyankomasi': 'Assin Nyankomasi',
        'akropong': 'Akropong',
        'tumfa': 'Tumfa',
        'kwabeng': 'Kwabeng',
        'assin atonsu': 'Assin Atonsu',
        'atonsu': 'Assin Atonsu',
        'awenare': 'Awenare',
        'abomosu': 'Abomosu',
        'mampamhwe': 'Mampamhwe',
        'assin brofoyedur': 'Assin Brofoyedur',
        'assin aponsie': 'Assin Aponsie',
        'banso': 'Banso',
        'mouso': 'Mouso',
        'amonom': 'Amonom',
        'asunafo': 'Asunafo',
        'apampatia': 'Apampatia',
        'assin wurakese': 'Assin Wurakese',
        'assin mesre nyame': 'Assin Mesre Nyame',
    }
    return mapping.get(name, name.title())

def clean_data(df):
    df = df.copy()
    df = df.drop(columns=[col for col in DROP_COLS if col in df.columns], errors='ignore')

    if 'prediction_time' in df.columns:
        df['prediction_time'] = pd.to_datetime(df['prediction_time'], errors='coerce')
        df['prediction_hour'] = df['prediction_time'].dt.hour
        df['prediction_month'] = df['prediction_time'].dt.month
        df = df.drop(columns=['prediction_time'], errors='ignore')

    if 'community' in df.columns:
        df['community'] = df['community'].apply(standardise_community)

    categorical_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
    if categorical_cols:
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    # Use configured numerical columns if provided, otherwise infer from dtypes
    numerical_cols = [col for col in NUMERICAL_COLS if col in df.columns] if NUMERICAL_COLS else df.select_dtypes(include=['number']).columns.tolist()
    if numerical_cols:
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    return df

def encode_features(df):
    df = df.copy()
    categorical_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def prepare_features(df):
    df = clean_data(df)

    # local import to prevent editor/linter unresolved import warnings

    target_map = {c: i for i, c in enumerate(TARGET_CLASSES)}
    df[TARGET_COL] = df[TARGET_COL].map(target_map).astype(int)

    df = encode_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )