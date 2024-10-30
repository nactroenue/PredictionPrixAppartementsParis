import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from config import (
    GeneralSettings,
    FeatureSelectionParams,
    EncodingFeatures,
    FeatureEngineering
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from category_encoders import TargetEncoder
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engineering import (
    calculate_vif,
    process_for_multicollinearity,
    create_interaction_terms,
    calculate_distance_to_landmarks,
    SpatialClusterAdder,
    scale_spatial_features
)
import joblib
import os
from typing import Tuple

def extract_coordinates(df):
    """
    Extrait la longitude et la latitude de la colonne 'geo_point_2d',
    en rendant ces coordonnées disponibles en tant que features séparées.
    
    Paramètres :
        df (pd.DataFrame): DataFrame avec une colonne 'geo_point_2d' contenant les informations de coordonnées.
        
    Returns:
        pd.DataFrame: DataFrame avec les colonnes 'latitude' et 'longitude' extraites.
    """
    if 'geo_point_2d' in df.columns:
        # Vérifier si 'geo_point_2d' est un dictionnaire ou une liste/tuple et extraire en conséquence
        if isinstance(df['geo_point_2d'].iloc[0], dict):
            df['longitude'] = df['geo_point_2d'].apply(lambda x: x.get('lon') if isinstance(x, dict) else np.nan)
            df['latitude'] = df['geo_point_2d'].apply(lambda x: x.get('lat') if isinstance(x, dict) else np.nan)
        elif isinstance(df['geo_point_2d'].iloc[0], (list, tuple)):
            df['latitude'] = df['geo_point_2d'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) >= 2 else np.nan)
            df['longitude'] = df['geo_point_2d'].apply(lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) >= 2 else np.nan)
        else:
            logging.warning("Format inattendu pour 'geo_point_2d'. Dictionnaire ou liste/tuple attendu.")
            df['longitude'] = np.nan
            df['latitude'] = np.nan
        # Supprimer la colonne d'origine 'geo_point_2d' après extraction
        df = df.drop(columns=['geo_point_2d'], errors='ignore')
    else:
        logging.warning("Colonne 'geo_point_2d' non trouvée dans le DataFrame. 'longitude' et 'latitude' ne seront pas créées.")
        df['longitude'] = np.nan
        df['latitude'] = np.nan
    
    return df

# Transformateur personnalisé pour le calcul des distances
class DistanceCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, center_coordinates: tuple):
        self.center_coordinates = center_coordinates

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X['distance_from_center'] = X.apply(
            lambda row: np.sqrt(
                (row['longitude'] - self.center_coordinates[0])**2 + 
                (row['latitude'] - self.center_coordinates[1])**2
            ),
            axis=1
        )
        return X

# Transformateur personnalisé pour les distances aux monuments
class LandmarkDistanceCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.landmarks = FeatureEngineering.LANDMARKS 

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return calculate_distance_to_landmarks(X, self.landmarks)

# Transformateur personnalisé pour la sélection de features (VIF)
class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 10.0, protected_features: list = None):
        self.threshold = threshold
        self.protected_features = protected_features if protected_features else []
        self.features_to_keep = []

    def fit(self, X, y=None):
        return process_for_multicollinearity(X, threshold=self.threshold, protected_features=self.protected_features)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features_to_keep + ['average_price']] if 'average_price' in X.columns else X[self.features_to_keep]

# Pipeline de sélection des features
def feature_selection_pipeline(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, n_features: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applique la méthode RFE (Elimination Récursive des Features) pour la sélection des features.
    
    Paramètres :
        X_train (pd.DataFrame): Features d'entraînement.
        y_train (pd.Series): Variable cible d'entraînement.
        X_test (pd.DataFrame): Features de test.
        n_features (int): Nombre de features à sélectionner.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Features sélectionnées pour l'entraînement et le test.
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=GeneralSettings.RANDOM_SEED)
    rfe = RFE(estimator=rf, n_features_to_select=n_features, step=10, n_jobs=-1)
    rfe.fit(X_train, y_train)
    selected_features = X_train.columns[rfe.support_]
    logging.info(f"Selected {len(selected_features)} features sélectionnées avec la méthode RFE.")
    return X_train[selected_features], X_test[selected_features]

def encode_binary_column(df):
    """
    Encode la colonne 'meuble_txt' en une colonne binaire 'meuble_binary' indiquant
    si l'appartement est meublé (1) ou non (0).
    """
    if 'meuble_txt' in df.columns:
        df['meuble_binary'] = df['meuble_txt'].apply(lambda x: 1 if x == 'meublé' else 0)
        # Supprimer la colonne 'meuble_txt' après l'encodage
        df = df.drop(columns=['meuble_txt'], errors='ignore')
    else:
        logging.warning("Colonne 'meuble_txt' non trouvée dans le DataFrame. 'meuble_binary' ne sera pas créée.")
        df['meuble_binary'] = 0
    return df

def create_target_column(df):
    """
    Ajoute une colonne cible 'prix_moyen' calculée comme la moyenne des colonnes de prix 'max' et 'min'.
    """
    # S'assurer que 'max' et 'min' sont des floats pour une moyenne correcte
    df['max'] = pd.to_numeric(df['max'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['average_price'] = (df['max'] + df['min']) / 2
    return df

def load_and_prepare_data(file_path: str, center_coordinates: tuple) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Charger, prétraiter et diviser les données en ensembles d'entraînement et de test en utilisant des Pipelines.
    
    Paramètres :
        file_path (str): Chemin vers le fichier de données JSON.
        center_coordinates (tuple): (longitude, latitude) du centre-ville
    
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Données d'entraînement et de test prétraitées.
    """
    if not os.path.exists(file_path):
        logging.error(f"Le fichier {file_path} n'existe pas.")
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    
    try:
        df = pd.read_json(file_path, orient="columns")
        df = extract_coordinates(df)  # S'assurer que les coordonnées sont extraites
        df = encode_binary_column(df)  # Convertir 'meuble_txt' en 'meuble_binary'
        df = create_target_column(df)  # Créer 'prix_moyen' comme variable cible
        logging.info(f"Loaded data with shape: {df.shape} and columns: {df.columns.tolist()}")
    except ValueError as e:
        logging.error(f"Erreur lors de la lecture du fichier JSON : {e}")
        raise

    # Vérifier que 'prix_moyen' a bien été créée
    if 'average_price' not in df.columns:
        logging.error("La variable cible 'prix_moyen' n'a pas pu être créée.")
        raise KeyError("La variable cible 'prix_moyen' est manquante.")

    # Diviser les données après extraction des coordonnées et encodage
    train_df, test_df = train_test_split(df, test_size=GeneralSettings.TEST_SIZE, random_state=GeneralSettings.RANDOM_SEED)
    logging.info(f"Forme des données d'entraînement : {train_df.shape}, Forme des données de test : {test_df.shape}")

    # Définir les pipelines de prétraitement pour les données numériques et catégorielles
    numerical_features = ['piece', 'meuble_binary', 'longitude', 'latitude']
    categorical_features = EncodingFeatures.ONE_HOT_ENCODING_FEATURES + EncodingFeatures.TARGET_ENCODING_FEATURES

    # Journaliser les colonnes disponibles dans df_entrainement pour vérifier que toutes les colonnes requises sont présentes
    logging.debug(f"Colonnes dans les données d'entraînement avant la configuration du pipeline : {train_df.columns.tolist()}")

    # Poursuivre avec la configuration du reste du pipeline et les transformations prévues
    numerical_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Pipeline catégoriel
    categorical_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    # Préprocesseur combiné
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='drop')

    full_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor)
    ])

    # Ajuster et transformer les données d'entraînement
    logging.info("Ajustement et transformation des données d'entraînement...")
    try:
        processed_train_array = full_pipeline.fit_transform(train_df)
        feature_names = numerical_features + list(full_pipeline.named_steps['preprocessing'].transformers_[1][1].get_feature_names_out(categorical_features))
        processed_train_df = pd.DataFrame(processed_train_array, columns=feature_names)
        logging.info(f"Forme des données d'entraînement traitées : {processed_train_df.shape}")
    except Exception as e:
        logging.error(f"Erreur lors de l'ajustement et de la transformation du pipeline sur les données d'entraînement : {e}")
        raise
    
    # Transformer les données de test
    logging.info("Transformation des données de test...")
    try:
        processed_test_array = full_pipeline.transform(test_df)
        processed_test_df = pd.DataFrame(processed_test_array, columns=feature_names)
        logging.info(f"Forme des données de test traitées : {processed_test_df.shape}")
    except Exception as e:
        logging.error(f"Erreur lors de la transformation du pipeline sur les données de test : {e}")
        raise
    
    # Séparer la variable cible
    y_train = train_df['average_price'].reset_index(drop=True)
    y_test = test_df['average_price'].reset_index(drop=True)
    logging.info("Séparation des features et de la variable cible.")
    
    return processed_train_df, y_train, processed_test_df, y_test