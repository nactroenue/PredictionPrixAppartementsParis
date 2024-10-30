# Prédiction des Prix des Appartements à Paris 🏡

## Objectif du Projet

L'objectif principal de ce projet est de développer des modèles de régression avancés pour prédire les prix des appartements à Paris. En utilisant des caractéristiques géographiques, de localisation et de propriétés, nous cherchons à comprendre les facteurs clés qui influencent les prix immobiliers et à fournir des prédictions précises pour aider les acheteurs, les vendeurs et les professionnels de l'immobilier.

## Description des Fichiers

**config.py**

Contient les paramètres globaux du projet, y compris les chemins des fichiers, les hyperparamètres des modèles et les configurations pour l'ingénierie des features et les visualisations.

<details><summary>Voir le code</summary>
  
```python

import os

# ============================================
#               Paramètres Généraux
# ============================================

class GeneralSettings:
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2

# ============================================
#                  Chemins Fichiers
# ============================================

class FilePaths:
    DATA_FILE_PATH = os.getenv('DATA_FILE_PATH', 'C:/Users/33766/Downloads/Appartment Price Prediction/apartment_data.json')
    MODEL_SAVE_PATHS = {
        'Scaler': 'models/scaler.joblib',
        'RandomForest': 'models/RandomForest_model.joblib',
        'GradientBoosting': 'models/GradientBoosting_model.joblib',
        'ExtraTrees': 'models/ExtraTrees_model.joblib',
        'XGBoost': 'models/XGBoost_model.joblib',
        'LightGBM': 'models/LightGBM_model.joblib',
        'MLP': 'models/MLP_model.joblib',
        'DNN': 'models/best_dnn_model.keras',
        'Linear': 'models/Linear_model.joblib',
        'DecisionTree': 'models/DecisionTree_model.joblib',
        'Ensemble': 'models/best_ensemble_model.joblib'
    }
    PROCESSED_DATA_PATHS = {
        'X_train': 'processed/X_train.joblib',
        'y_train': 'processed/y_train.joblib',
        'X_test': 'processed/X_test.joblib',
        'y_test': 'processed/y_test.joblib'
    }

# ============================================
#        Grilles d'Hyperparamètres pour les Modèles
# ============================================

class HyperparameterGrids:
    EXTRA_TREES = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    RANDOM_FOREST = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    GRADIENT_BOOSTING = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.7, 0.8, 0.9]
    }
    XGBOOST = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    LIGHTGBM = {
        'objective': ['regression'],
        'metric': ['rmse'],
        'boosting_type': ['gbdt'],
        'max_bin': [128, 255, 512],
        'num_leaves': [31, 61, 91],
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0.0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.0, 0.1, 0.5, 1.0],
        'min_split_gain': [0.0, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500, 1000]
    }

# ============================================
#         Poids de l'Ensemble pour le VotingRegressor
# ============================================

class EnsembleSettings:
    WEIGHTS = [0.4, 0.35, 0.15, 0.10]  # [RandomForest, GradientBoosting, XGBoost, LightGBM]

# ============================================
#        Paramètres de Sélection des Features
# ============================================

class FeatureSelectionParams:
    THRESHOLD = 0.001
    CORR_THRESHOLD = 0.9
    N_FEATURES = 30

# ============================================
#               Configuration de la Journalisation
# ============================================

class LoggingConfig:
    LEVEL = 'INFO'
    FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ============================================
#            Paramètres du Réseau de Neurones
# ============================================

class NeuralNetworkSettings:
    EPOCHS = 500
    BATCH_SIZE = 32
    PATIENCE = 50
    LEARNING_RATE = [0.01, 0.005, 0.001]

# ============================================
#        Paramètres d'Ingénierie des Features
# ============================================

class FeatureEngineering:
    SPATIAL_CLUSTERS = 10
    LANDMARKS = {
        'Eiffel_Tower': (2.2945, 48.8584),
        'Louvre_Museum': (2.3364, 48.8606),
        'Notre_Dame': (2.3499, 48.8530)
    }

# ============================================
#        Encodage des Features Catégorielles
# ============================================

class EncodingFeatures:
    ONE_HOT_ENCODING_FEATURES = ['ville']
    TARGET_ENCODING_FEATURES = ['nom_quartier']

# ============================================
#        Définition des Features d'Interaction
# ============================================

class InteractionFeatures:
    DEFINITIONS = {
        'longitude_latitude_interaction': ['longitude', 'latitude'],
        'distance_city_center_piece_interaction': ['distance_from_center', 'piece'],
        'distance_to_Eiffel_Tower_bathroom_interaction': ['distance_to_Eiffel_Tower', 'bathroom'],
        'distance_to_Louvre_Museum_bathroom_interaction': ['distance_to_Louvre_Museum', 'bathroom']
    }

# ============================================
#        Gestion des Formes Géographiques
# ============================================

class GeoShapeConfig:
    INCLUDE_GEO_SHAPE = False
    EXCLUDED_FEATURES = ['ref']
```
</details>

**2. data_preparation.py**

Script pour charger, préparer et diviser les données. Inclut l'extraction des coordonnées géographiques, l'encodage des variables catégorielles et la création de la colonne cible average_price.

<details><summary>Voir le code</summary>
  
```python

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
        df['meuble_binary'] = 0  # Attribuer une valeur par défaut ou gérer en conséquence
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
```

</details>

**3. feature_engineering.py**

Gère l'ingénierie des features, notamment le calcul des distances aux monuments, la création d'interactions entre les variables et la réduction de la multicolinéarité avec des méthodes comme RFE.classes.

<details><summary>Voir le code</summary>

```python

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

def calculate_vif(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Calculer le VIF pour chaque feature de la liste."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif_data

def process_for_multicollinearity(
    df: pd.DataFrame, 
    threshold: float = 10.0, 
    protected_features: list = None
) -> pd.DataFrame:
    """
    Supprimer les features multicolinéaires en fonction du VIF.
    
    Paramètres:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Seuil de VIF pour supprimer les features.
        protected_features (list): Features à protéger contre la suppression.
    
    Returns:
        pd.DataFrame: DataFrame avec une multicolinéarité réduite.
    """
    if protected_features is None:
        protected_features = []
    
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    logging.info(f"Colonnes numériques avant calcul du VIF : {df_numeric.columns.tolist()}")

    # Exclure la variable cible si présente
    target = 'average_price'
    if target in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=[target])
        logging.info(f"Exclusion de '{target}' du calcul du VIF.")

    # Gérer les valeurs NaN ou infinies
    df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_numeric.fillna(df_numeric.mean(), inplace=True)

    # Supprimer les colonnes à variance zéro
    zero_var_cols = df_numeric.columns[df_numeric.var() == 0].tolist()
    if zero_var_cols:
        df_numeric.drop(columns=zero_var_cols, inplace=True)
        logging.info(f"Colonnes à variance zéro supprimées : {zero_var_cols}")

    if df_numeric.empty:
        logging.error("DataFrame vide après les étapes de nettoyage.")
        return df_numeric

    features = df_numeric.columns.tolist()
    dropped_features = []

    while True:
        vif_df = calculate_vif(df_numeric, features)
        max_vif = vif_df['VIF'].max()
        if max_vif > threshold:
            feature_to_drop = vif_df.loc[vif_df['VIF'].idxmax(), 'Feature']
            if feature_to_drop in protected_features:
                logging.info(f"Feature protégée '{feature_to_drop}' avec un VIF élevé ({max_vif}) et ne sera pas supprimée.")
                break
            features.remove(feature_to_drop)
            dropped_features.append(feature_to_drop)
            logging.info(f"Suppression de '{feature_to_drop}' avec VIF={max_vif:.2f}")
        else:
            break

    logging.info(f"Features supprimées en raison d'un VIF élevé : {dropped_features}")
    logging.info(f"Features finales après traitement du VIF : {features}")
    return df[features + [target]] if target in df.columns else df[features]

def create_interaction_terms(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Créer des termes d'interaction entre les features spécifiées.

    Paramètres :
        df (pd.DataFrame): Input DataFrame.
        features (list): Liste des noms de features à interagir.

    Returns:
        pd.DataFrame: DataFrame avec les termes d'interaction ajoutés.
    """
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            feature_name = f"{features[i]}_x_{features[j]}"
            df[feature_name] = df[features[i]] * df[features[j]]
            logging.debug(f"Création du terme d'interaction : {feature_name}")
    return df

def calculate_distance_to_landmarks(df: pd.DataFrame, landmarks: dict) -> pd.DataFrame:
    """
    Calculer les distances entre chaque appartement et des monuments prédéfinis.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame avec les colonnes 'latitude' et 'longitude'.
        landmarks (dict): Dictionnaire de monuments avec les noms comme clés et les tuples (latitude, longitude) comme valeurs.
    
    Returns:
        pd.DataFrame: DataFrame avec de nouvelles colonnes de distance pour chaque monument.
    """
    for landmark, coords in landmarks.items():
        column_name = f"distance_au_{landmark}"
        df[column_name] = df.apply(
            lambda row: geodesic((row['latitude'], row['longitude']), coords).kilometers 
            if pd.notnull(row['latitude']) and pd.notnull(row['longitude']) else np.nan,
            axis=1
        )
        logging.debug(f"Calcul de la distance au {landmark}")
    return df

class SpatialClusterAdder(BaseEstimator, TransformerMixin):
    """
    Transformateur personnalisé pour ajouter des clusters spatiaux à une DataFrame en utilisant le clustering KMeans.
    
    Paramètres :
        n_clusters (int): Nombre de clusters pour KMeans.
        random_state (int): Graine aléatoire pour la reproductibilité.
    """
    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def fit(self, X, y=None):
        if 'longitude' in X.columns and 'latitude' in X.columns:
            coords = X[['longitude', 'latitude']]
            self.kmeans.fit(coords)
        else:
            raise ValueError("Colonnes 'longitude' et 'latitude' attendues pour le clustering.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        coords = X[['longitude', 'latitude']]
        X = X.copy()
        X['spatial_cluster'] = self.kmeans.predict(coords)
        logging.debug(f"Ajout de 'cluster_spatial' avec {self.n_clusters} clusters")
        return X

def scale_spatial_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Normaliser les features spatiales spécifiées en utilisant StandardScaler.
    
    Paramètres :
        df (pd.DataFrame): Input DataFrame.
        features (list): Liste des noms de features à normaliser.
    
    Returns:
        pd.DataFrame: DataFrame avec les features normalisées.
    """
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    logging.debug(f"Normalisation des features spatiales : {features}")
    return df

```

</details>

**4. model_training.py**

Script pour entraîner et évaluer les modèles de régression, y compris le réglage des hyperparamètres et l'utilisation d'un modèle d'ensemble.

<details><summary>Voir le code</summary>

```python

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    VotingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import joblib
import logging
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from utils import Metrics, Visualization, TensorFlowMetrics
from config import (
    GeneralSettings,
    FilePaths,
    HyperparameterGrids,
    NeuralNetworkSettings,
    EnsembleSettings
)
from typing import Tuple, Dict, Any
from visualization import Visualization
import os


def train_and_evaluate_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Entraîner et évaluer plusieurs modèles de régression avec réglage des hyperparamètres.

    Paramètres:
        X_train (pd.DataFrame): Caractéristiques d'entraînement.
        y_train (pd.Series): Cible d'entraînement.
        X_test (pd.DataFrame): Caractéristiques de test.
        y_test (pd.Series): Cible de test.

    Returns:
        Tuple[Dict[str, float], Dict[str, np.ndarray]]: Scores RMSE et prédictions des modèles.
    """
    logging.info("Début de l'entraînement et de l'évaluation des modèles...")

    # Initialiser et sauvegarder le scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, FilePaths.MODEL_SAVE_PATHS['Scaler'])
    logging.info(f"Scaler ajusté et sauvegardé à {FilePaths.MODEL_SAVE_PATHS['Scaler']}.")



    # Créer des caractéristiques polynomiales
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    feature_names = poly.get_feature_names_out(input_features=X_train.columns)
    logging.info("Caractéristiques polynomiales créées.")

    # Définir les modèles
    models = {
        'Linear': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(random_state=GeneralSettings.RANDOM_SEED),
        'RandomForest': RandomForestRegressor(random_state=GeneralSettings.RANDOM_SEED),
        'GradientBoosting': GradientBoostingRegressor(random_state=GeneralSettings.RANDOM_SEED),
        'ExtraTrees': ExtraTreesRegressor(random_state=GeneralSettings.RANDOM_SEED),
        'XGBoost': xgb.XGBRegressor(random_state=GeneralSettings.RANDOM_SEED, objective='reg:squarederror'),
        'LightGBM': lgb.LGBMRegressor(random_state=GeneralSettings.RANDOM_SEED),
        'MLP': MLPRegressor(random_state=GeneralSettings.RANDOM_SEED, max_iter=500)
    }

    # Grilles d'hyperparamètres
    param_grids = {
        'RandomForest': HyperparameterGrids.RANDOM_FOREST,
        'GradientBoosting': HyperparameterGrids.GRADIENT_BOOSTING,
        'ExtraTrees': HyperparameterGrids.EXTRA_TREES,
        'XGBoost': HyperparameterGrids.XGBOOST,
        'LightGBM': HyperparameterGrids.LIGHTGBM
    }

    model_rmse_scores = {}
    model_predictions = {}
    trained_models = {}

    # Créer le répertoire de graphiques
    plots_dir = 'C:/Users/33766/models/plots'
    os.makedirs(plots_dir, exist_ok=True)
    

    # Entraîner les modèles avec réglage des hyperparamètres
    for name, model in models.items():
        logging.info(f"Entraînement du modèle: {name}")
        if name in param_grids:
            best_params, best_model = tune_model(model, param_grids[name], X_train_scaled, y_train)
            logging.info(f"Meilleurs paramètres pour {name}: {best_params}")
        else:
            # Validation croisée pour les modèles sans hyperparamètres
            cv = KFold(n_splits=5, shuffle=True, random_state=GeneralSettings.RANDOM_SEED)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
            mean_rmse = -cv_scores.mean()
            std_rmse = cv_scores.std()
            logging.info(f"{name} CV RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
            model.fit(X_train_scaled, y_train)
            best_model = model

        # Prédictions et évaluation
        y_pred = best_model.predict(X_test_scaled)
        test_rmse = Metrics.rmse(y_test, y_pred)
        model_rmse_scores[name] = test_rmse
        model_predictions[name] = y_pred
        logging.info(f"RMSE du modèle {name} sur le test: {test_rmse:.4f}")

        # Tracer les prédictions vs réel
        pred_plot_path = os.path.join(plots_dir, f"{name}_predictions.png")
        Visualization.plot_predictions(y_test, y_pred, name, save_path=pred_plot_path)

        # Tracer les résidus
        resid_plot_path = os.path.join(plots_dir, f"{name}_residuals.png")
        Visualization.plot_residuals(y_test, y_pred, name, save_path=resid_plot_path)

        # Tracer l'importance des caractéristiques si applicable
        if hasattr(best_model, 'feature_importances_') or hasattr(best_model, "coef_"):
            save_path = os.path.join(plots_dir, f"{name}_feature_importance.png")
            Visualization.plot_feature_importance(
                best_model,
                feature_names,
                name,
                save_path=save_path,
                top_n=10
            )

        # Sauvegarder le modèle
        joblib.dump(best_model, FilePaths.MODEL_SAVE_PATHS[name])
        logging.info(f"{name} model saved to {FilePaths.MODEL_SAVE_PATHS[name]}.")

        trained_models[name] = best_model

    # Entraîner le réseau de neurones profond (DNN)
    dnn_rmse, dnn_preds, dnn_model = build_and_train_dnn(
        X_train_poly, y_train, X_test_poly, y_test, input_shape=X_train_poly.shape[1]
    )
    model_rmse_scores['DNN'] = dnn_rmse
    model_predictions['DNN'] = dnn_preds
    dnn_model.save(FilePaths.MODEL_SAVE_PATHS['DNN'])
    logging.info(f"Modèle DNN sauvegardé à {FilePaths.MODEL_SAVE_PATHS['DNN']}.")

    # Tracer les prédictions vs réel pour DNN
    dnn_pred_plot_path = os.path.join(plots_dir, "DNN_predictions.png")
    Visualization.plot_predictions(y_test, dnn_preds, "DNN", save_path=dnn_pred_plot_path)

    # Tracer les résidus pour DNN
    dnn_resid_plot_path = os.path.join(plots_dir, "DNN_residuals.png")
    Visualization.plot_residuals(y_test, dnn_preds, "DNN", save_path=dnn_resid_plot_path)

    # Entraîner l'ensemble
    ensemble_rmse, ensemble_preds, ensemble_model = build_ensemble_model(trained_models, X_train_scaled, y_train, X_test_scaled, y_test)
    model_rmse_scores['Ensemble'] = ensemble_rmse
    model_predictions['Ensemble'] = ensemble_preds
    joblib.dump(ensemble_model, FilePaths.MODEL_SAVE_PATHS['Ensemble'])
    logging.info(f"Modèle Ensemble sauvegardé à {FilePaths.MODEL_SAVE_PATHS['Ensemble']}.")

    # Tracer les prédictions vs réel pour l'ensemble
    ensemble_pred_plot_path = os.path.join(plots_dir, "Ensemble_predictions.png")
    Visualization.plot_predictions(y_test, ensemble_preds, "Ensemble", save_path=ensemble_pred_plot_path)

    # Tracer les résidus pour l'ensemble
    ensemble_resid_plot_path = os.path.join(plots_dir, "Ensemble_residuals.png")
    Visualization.plot_residuals(y_test, ensemble_preds, "Ensemble", save_path=ensemble_resid_plot_path)

    # Tracer l'importance des caractéristiques pour l'ensemble si applicable
    if hasattr(ensemble_model, 'feature_importances_') or hasattr(ensemble_model, "coef_"):
        ensemble_feat_imp_path = os.path.join(plots_dir, "Ensemble_feature_importance.png")
        logging.info(f"Tentative de tracé de l'importance des caractéristiques pour l'ensemble")
        Visualization.plot_feature_importance(
            ensemble_model,
            feature_names,
            "Ensemble",
            save_path=ensemble_feat_imp_path,
            top_n=10
        )
    else:
        logging.info(f"Pas d'importance des caractéristiques ou de coefficients à tracer pour l'ensemble.")

    # Enregistrer les scores RMSE finaux
    logging.info("Scores RMSE finaux des modèles:")
    for model_name, rmse_score in model_rmse_scores.items():
        logging.info(f"{model_name}: RMSE = {rmse_score:.4f}")

     # Générer le rapport automatisé
    try:
        Visualization.generate_automated_report(
            rmse_scores=model_rmse_scores,
            model_predictions=model_predictions,
            y_test=y_test,
            report_path="C:/Users/33766/models/automated_model_report.html",
            feature_names=feature_names.tolist(),
        )
        logging.info("Rapport automatisé généré avec succès.")
    except Exception as e:
        logging.error(f"Échec de la génération du rapport automatisé: {e}")

    return model_rmse_scores, model_predictions


def tune_model(model: Any, param_grid: dict, X_train: np.ndarray, y_train: pd.Series) -> Tuple[dict, Any]:
    """
    Effectuer le réglage des hyperparamètres à l'aide de RandomizedSearchCV.

    Paramètres:
        model (Any): Estimateur de Scikit-learn
        param_grid (dict): Grille d'hyperparamètres.
        X_train (np.ndarray): Caractéristiques d'entraînement.
        y_train (pd.Series): Cible d'entraînement.

    Returns:
        Tuple[dict, Any]: Meilleurs paramètres et meilleur estimateur.
    """
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        scoring='neg_root_mean_squared_error',
        cv=5,
        random_state=GeneralSettings.RANDOM_SEED,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    logging.info(f"RandomizedSearchCV terminé pour {model.__class__.__name__}")
    return search.best_params_, search.best_estimator_


def build_and_train_dnn(X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, y_test: pd.Series, input_shape: int) -> Tuple[float, np.ndarray, keras.Model]:
    """
    Construction, entraînement et évaluation un réseau de neurones profond (DNN).

    Paramètres:
        X_train (np.ndarray): Caractéristiques d'entraînement avec caractéristiques polynomiales.
        y_train (pd.Series): Cible d'entraînement.
        X_test (np.ndarray): Caractéristiques de test avec caractéristiques polynomiales.
        y_test (pd.Series): Cible de test.
        input_shape (int): Nombre de caractéristiques d'entrée.

    Retourne:
        Tuple[float, np.ndarray, keras.Model]: Meilleur RMSE, meilleures prédictions et meilleur modèle.
    """
    best_rmse = float('inf')
    best_model = None
    best_lr = None
    best_y_pred = None

    for lr in NeuralNetworkSettings.LEARNING_RATE:
        logging.info(f"Entraînement du DNN avec un taux d'apprentissage de: {lr}")
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=TensorFlowMetrics.root_mean_squared_error
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=NeuralNetworkSettings.PATIENCE,
            restore_best_weights=True
        )
        lr_scheduler = keras.callbacks.LearningRateScheduler(adjusted_scheduler)

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=NeuralNetworkSettings.EPOCHS,
            batch_size=NeuralNetworkSettings.BATCH_SIZE,
            callbacks=[early_stopping, lr_scheduler],
            verbose=0
        )

        y_pred = model.predict(X_test).flatten()
        current_rmse = Metrics.rmse(y_test, y_pred)
        logging.info(f"DNN avec LR={lr} a obtenu un RMSE={current_rmse:.4f}")

        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_model = model
            best_lr = lr
            best_y_pred = y_pred

    logging.info(f"Meilleur taux d'apprentissage DNN: {best_lr} avec RMSE: {best_rmse:.4f}")
    return best_rmse, best_y_pred, best_model


def adjusted_scheduler(epoch: int, current_lr: float) -> float:
    """
    Ajustement le taux d'apprentissage après un certain nombre d'époques.

    Paramètres:
        epoch (int): Numéro de l'époque actuelle.
        current_lr (float): Taux d'apprentissage actuel.

    Retourne:
        float: Taux d'apprentissage mis à jour.
    """
    if epoch >= 30:
        return current_lr * np.exp(-0.005 * (epoch - 30))
    return current_lr


def build_ensemble_model(trained_models: Dict[str, Any], X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, y_test: pd.Series) -> Tuple[float, np.ndarray, VotingRegressor]:
    """
    Construire et évaluer un modèle d'ensemble à l'aide de VotingRegressor.

    Paramètres:
        trained_models (Dict[str, Any]): Dictionnaire de modèles entraînés.
        X_train (np.ndarray): Caractéristiques d'entraînement mises à l'échelle.
        y_train (pd.Series): Cible d'entraînement.
        X_test (np.ndarray): Caractéristiques de test mises à l'échelle.
        y_test (pd.Series): Cible de test.

    Retourne:
        Tuple[float, np.ndarray, VotingRegressor]: RMSE de l'ensemble, prédictions et modèle d'ensemble.
    """
    ensemble_estimators = [
        ('rf', trained_models.get('RandomForest')),
        ('gb', trained_models.get('GradientBoosting')),
        ('xgb', trained_models.get('XGBoost')),
        ('lgbm', trained_models.get('LightGBM'))
    ]
    
    # Supprimer les estimateurs None si présents
    ensemble_estimators = [est for est in ensemble_estimators if est[1] is not None]
    
    if not ensemble_estimators:
        logging.error("Aucun modèle valide disponible pour l'ensemble.")
        raise ValueError("Aucun modèle valide disponible pour l'ensemble.")
    
    ensemble_model = VotingRegressor(
        estimators=ensemble_estimators,
        weights=EnsembleSettings.WEIGHTS
    )
    
    ensemble_model.fit(X_train, y_train)
    ensemble_pred = ensemble_model.predict(X_test)
    ensemble_rmse = Metrics.rmse(y_test, ensemble_pred)
    logging.info(f"Ensemble Model RMSE: {ensemble_rmse:.4f}")
    
    return ensemble_rmse, ensemble_pred, ensemble_model

```
</details>


**5. visualization.py**

Contient des fonctions pour générer des graphiques et des rapports automatisés, tels que l'importance des caractéristiques, les prédictions vs réel et la distribution des résidus.

<details> <summary>Voir le code</summary>

```python

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from jinja2 import Environment, FileSystemLoader
import shutil


class Visualization:
    """Classe utilitaire pour créer des visualisations, générer des rapports et configurer des tableaux de bord."""
    
    @staticmethod
    def plot_feature_importance(model, feature_names, model_name, save_path, top_n=10):
        """Trace et enregistre les N meilleures importances ou coefficients des caractéristiques avec des valeurs affichées."""
        if hasattr(model, "feature_importances_"):
            # Pour les modèles basés sur les arbres avec feature_importances_
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_n:][::-1]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]

            plt.figure(figsize=(12, 8))
            plt.title(f"Top {top_n} Importances des Caractéristiques - {model_name}")
            bars = plt.barh(range(top_n), top_importances, align="center")
            plt.yticks(range(top_n), top_features)
            plt.xlabel("Importance")
            plt.tight_layout()

            # Annoter chaque barre avec la valeur de l'importance, ajustée pour les grandes valeurs
            for bar, importance in zip(bars, top_importances):
                if importance > 1:
                    # Affichage des grandes valeurs en notation scientifique si nécessaire
                    formatted_value = f'{importance:.0f}' if importance < 1e3 else f'{importance:.2e}'
                else:
                    formatted_value = f'{importance:.3f}'
                
                plt.text(
                    bar.get_width(), bar.get_y() + bar.get_height()/2,
                    formatted_value, va='center', ha='left'
                )

            plt.savefig(save_path)
            plt.close()
            logging.info(f"Graphique d'importance des caractéristiques pour {model_name} dans {save_path}. Affichage des {top_n} meilleures caractéristiques.")
        
        elif hasattr(model, "coef_"):
            logging.debug(f"Génération du graphique des coefficients pour {model_name}")
            # Pour les modèles linéaires avec coef_
            coefficients = model.coef_
            if coefficients is None or len(coefficients) == 0:
                logging.warning(f"Aucun coefficient trouvé pour {model_name}. Passage du graphique.")
                return
            indices = np.argsort(np.abs(coefficients))[-top_n:][::-1]  # Trier par valeur absolue des coefficients
            top_features = [feature_names[i] for i in indices]
            top_coefficients = coefficients[indices]

            plt.figure(figsize=(12, 8))
            plt.title(f"Top {top_n} Coefficients des Caractéristiques - {model_name}")
            bars = plt.barh(range(top_n), top_coefficients, align="center")
            plt.yticks(range(top_n), top_features)
            plt.xlabel("Valeur du Coefficient")
            plt.tight_layout()

            for bar, coef in zip(bars, top_coefficients):
                formatted_coef = f'{coef:.2f}' if np.abs(coef) < 1e3 else f'{coef:.2e}'
                plt.text(
                    bar.get_width(), bar.get_y() + bar.get_height()/2,
                    formatted_coef, va='center', ha='left'
                )

            plt.savefig(save_path)
            plt.close()
            logging.info(f"Graphique des coefficients pour {model_name} enregistré dans {save_path}. Affichage des {top_n} meilleurs coefficients.")
        
        else:
            logging.warning(f"{model_name} ne prend pas en charge l'affichage des importances ou des coefficients des caractéristiques.")
    
    @staticmethod
    def generate_automated_report(
        rmse_scores, model_predictions, y_test, report_path="C:/Users/33766/Downloads/Appartment Price Prediction/report_template.html", feature_names=None
    ):
        """Génère un rapport HTML résumant les performances des modèles."""
        if not os.path.exists('templates'):
            os.makedirs('templates')
        
        # Définir le répertoire où les graphiques sont enregistrés
        plots_dir = 'C:/Users/33766/models/plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Initialiser l'environnement Jinja2
        env = Environment(loader=FileSystemLoader('C:/Users/33766/Downloads/Appartment Price Prediction'))
        template = env.get_template('report_template.html')  # Assurez-vous que ce modèle existe dans le répertoire courant
        
        models = rmse_scores.keys()
        plot_paths = {}
        
        for model in models:
            plot_paths[model] = {}
            # Définir les chemins relatifs pour différents graphiques
            pred_plot = os.path.join(plots_dir, f"{model}_predictions.png")
            resid_plot = os.path.join(plots_dir, f"{model}_residuals.png")
            feat_imp_plot = os.path.join(plots_dir, f"{model}_feature_importance.png")
            
            
            if os.path.exists(pred_plot):
                plot_paths[model]['predictions'] = pred_plot
            else:
                logging.warning(f"Graphique des prédictions pour {model} introuvable à {pred_plot}.")
                plot_paths[model]['predictions'] = ''
            
            if os.path.exists(resid_plot):
                plot_paths[model]['residuals'] = resid_plot
            else:
                logging.warning(f"Residuals plot for {model} not found at {resid_plot}.")
                plot_paths[model]['residuals'] = ''
            
            
            if os.path.exists(feat_imp_plot):
                plot_paths[model]['feature_importance'] = feat_imp_plot
            else:
                plot_paths[model]['feature_importance'] = None
        
        
        html_content = template.render(
            rmse_scores=rmse_scores,
            models=models,
            plot_paths=plot_paths
        )
        
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        logging.info(f"Rapport automatisé généré à {report_path}")
    
    @staticmethod
    def plot_predictions(y_true, y_pred, model_name, save_path):
        """Trace et enregistre les prix prédits par rapport aux prix réels pour un modèle."""
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, edgecolor='k')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
        plt.title(f'{model_name} Prix Prévus vs. Prix Réels')
        plt.xlabel('Prix Réels')
        plt.ylabel('Prix Prévus')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Graphique des prix prévus vs. réels pour {model_name} enregistré dans {save_path}.")
    
    @staticmethod
    def plot_residuals(y_true, y_pred, model_name, save_path):
        """Trace et enregistre la distribution des résidus pour un modèle."""
        residuals = y_true - y_pred
        plt.figure(figsize=(12, 8))
        sns.histplot(residuals, bins=50, kde=True, color='salmon')
        plt.title(f'{model_name} Distribution des Résidus')
        plt.xlabel('Résidus')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Graphique de distribution des résidus pour {model_name} enregistré dans {save_path}.")

```

</details>

**6. utils.py**

Fonctions utilitaires pour le calcul des métriques de régression (RMSE, MAE, R²) et pour la visualisation des performances des modèles.
                                                                  
<details> <summary>Voir le code</summary>

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Callable, Union

class Metrics:
    """Classe utilitaire pour les métriques de régression courantes."""
    
    @staticmethod
    def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Calculer l'erreur quadratique moyenne (RMSE).
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Calculer l'erreur absolue moyenne (MAE).
        """
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def r2(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Calculer le coefficient de détermination (R²).
        """
        return r2_score(y_true, y_pred)

    @staticmethod
    def custom_metric(y_true: pd.Series, y_pred: np.ndarray, metric_fn: Callable) -> float:
        """
        Calculer une métrique personnalisée fournie sous forme de fonction.
        
        Paramètres:
            y_true (pd.Series): Valeurs cibles réelles.
            y_pred (np.ndarray): Valeurs cibles prédites.
            metric_fn (Callable): Fonction prenant y_vrai et y_pred et renvoyant une métrique.
            
        Returns:
            float: Résultat de la métrique.
        """
        return metric_fn(y_true, y_pred)

class Visualization:
    """Classe utilitaire pour tracer les performances du modèle et l'importance des features."""
    
    @staticmethod
    def plot_feature_importance(model, feature_names: List[str], model_name: str, save_path: str = None):
        """
        Tracer l'importance des features pour les modèles basés sur les arbres.
        """
        if not hasattr(model, 'feature_importances_'):
            logging.warning(f"{model_name} n'a pas l'attribut 'feature_importances_'.")
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title(f"{model_name} Feature Importances")
        plt.bar(range(len(importances)), importances[indices], color='steelblue')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Graphique d'importance des features pour {model_name} enregistré à {save_path}")
        else:
            plt.show()

    @staticmethod
    def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, model_name: str, save_path: str = None):
        """
        Tracer la distribution des résidus pour les prédictions.
        """
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=50, color='lightcoral', edgecolor='k')
        plt.xlabel('Residuals')
        plt.title(f'Distribution des Résidus - {model_name}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Graphique de distribution des résidus pour {model_name} enregistré à {save_path}")
        else:
            plt.show()

    @staticmethod
    def plot_model_performance(rmse_scores: dict, save_path: str = None):
        """
        Tracer les scores RMSE pour différents modèles.
        """
        if not rmse_scores:
            logging.warning("Aucun score RMSE fourni pour le tracé.")
            return

        models = list(rmse_scores.keys())
        rmse_values = list(rmse_scores.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, rmse_values, color='skyblue')
        plt.xlabel('Models')
        plt.ylabel('RMSE')
        plt.title('Comparaison des RMSE des Modèles')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Annoter les barres avec les valeurs RMSE
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Graphique de performance des modèles enregistré à {save_path}")
        else:
            plt.show()

class TensorFlowMetrics:
    """Classe utilitaire pour les métriques compatibles avec TensorFlow."""
    
    @staticmethod
    def root_mean_squared_error(y_true, y_pred) -> tf.Tensor:
        """
        Fonction de perte RMSE compatible avec TensorFlow.
        """
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

```

</details>

**7. main.py**

Script principal pour exécuter le pipeline de prédiction du prix des appartements.

<details> <summary>Voir le code</summary>

```python

import logging
import sys
import pandas as pd
from data_preparation import load_and_prepare_data
from model_training import train_and_evaluate_models
from config import FilePaths, LoggingConfig, FeatureSelectionParams
from utils import Metrics
from visualization import Visualization
import warnings

warnings.filterwarnings("ignore", message="main thread is not in main loop")

def setup_logging():
    """
    Configurer les paramètres de journalisation.
    """
    logging.basicConfig(
        level=getattr(logging, LoggingConfig.LEVEL),
        format=LoggingConfig.FORMAT,
        handlers=[
            logging.FileHandler("prediction_prix_appartements.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("La journalisation est configurée.")

def process_data(X: pd.DataFrame, threshold: float, protected_features: list) -> pd.DataFrame:
    """
    Traiter les données en appliquant un seuil aux features protégés.
    """
    processed_df = X.copy()
    # Appliquer un seuil uniquement aux features non-binaires existantes
    for feature in protected_features:
        if feature in processed_df.columns and processed_df[feature].nunique() > 2:
            processed_df[feature] = processed_df[feature].apply(lambda x: 1 if x > threshold else 0)
            logging.debug(f"Seuil appliqué à la feature protégée : {feature}")
        else:
            logging.info(f"Aucun seuil appliqué à '{feature}' car elle est déjà binaire ou non présente.")
    return processed_df

def main():
    """
    Fonction principale pour exécuter le pipeline de prédiction du prix des appartements.
    """
    setup_logging()
    logging.info("Démarrage du pipeline de prédiction de prix des appartements...")

    # Définir les coordonnées du centre-ville
    city_center = (2.3522, 48.8566)  # Centre-ville de Paris (longitude, latitude)

    # Charger et prétraiter les données
    try:
        X_train, y_train, X_test, y_test = load_and_prepare_data(FilePaths.DATA_FILE_PATH, center_coordinates=city_center)
        logging.info(f"Données chargées et divisées : X_train={X_train.shape}, X_test={X_test.shape}")
    except FileNotFoundError as e:
        logging.error(f"Échec du chargement des données : {e}")
        sys.exit(1)
    except KeyError as e:
        logging.error(f"Échec du prétraitement des données : {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Erreur inattendue lors du chargement des données : {e}")
        sys.exit(1)

    # Définir les features protégées et le seuil depuis la config ou les paramètres
    protected_features = ['meuble_binary']
    threshold = FeatureSelectionParams.THRESHOLD

    # Traiter les données d'entraînement
    try:
        processed_train_df = process_data(X_train, threshold=threshold, protected_features=protected_features)
        logging.info(f"Forme des données d'entraînement traitées : {processed_train_df.shape}")
    except Exception as e:
        logging.error(f"Erreur lors du traitement des données d'entraînement : {e}")
        sys.exit(1)

    # Traiter les données de test
    try:
        processed_test_df = process_data(X_test, threshold=threshold, protected_features=protected_features)
        logging.info(f"Forme des données de test traitées : {processed_test_df.shape}")
    except Exception as e:
        logging.error(f"Erreur lors du traitement des données de test : {e}")
        sys.exit(1)

    # Entraîner et évaluer les modèles
    try:
        model_rmse_scores, model_predictions = train_and_evaluate_models(
            processed_train_df, y_train, processed_test_df, y_test
        )
    except Exception as e:
        logging.error(f"Échec de l'entraînement et de l'évaluation des modèles : {e}")
        sys.exit(1)

    # Journaliser les scores RMSE finaux
    logging.info("Scores RMSE finaux des modèles :")
    for model, score in model_rmse_scores.items():
        logging.info(f"{model}: RMSE = {score:.4f}")

    logging.info("Pipeline de prédiction de prix des appartements terminé avec succès.")

    # Générer un rapport automatisé
    try:
        Visualization.generate_automated_report(
            rmse_scores=model_rmse_scores,
            model_predictions=model_predictions,
            y_test=y_test,
            report_path="models/automated_model_report.html",
            feature_names=processed_train_df.columns.tolist()  # Utiliser les noms des features des données d'entraînement traitées
        )
        logging.info("Rapport automatisé généré avec succès.")
    except Exception as e:
        logging.error(f"Échec de la génération du rapport automatisé : {e}")

if __name__ == "__main__":
    main()

```
</details>

**8. automated_model_report.html**

Modèle HTML utilisé pour générer le rapport automatisé présentant les performances des modèles.

<details> <summary>Voir le code</summary>

```html

<!DOCTYPE html>
<html>
<head>
    <title>Rapport Automatisé de Performance des Modèles</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2, h3 { color: #333; }
        table { width: 60%; border-collapse: collapse; margin-bottom: 40px; }
        th, td { border: 1px solid #ccc; padding: 12px; text-align: center; }
        th { background-color: #f4f4f4; }
        img { max-width: 100%; height: auto; margin-bottom: 40px; }
        .section { margin-bottom: 60px; }
    </style>
</head>
<body>
    <h1>Rapport Automatisé de Performance des Modèles</h1>

    <!-- Section 1: RMSE Scores -->
    <div class="section">
        <h2>1. Scores RMSE</h2>
        <table>
            <tr>
                <th>Modèle</th>
                <th>RMSE</th>
            </tr>
            
            <tr>
                <td>Linéaire</td>
                <td>1.8928</td>
            </tr>
            
            <tr>
                <td>Arbre de Décision</td>
                <td>1.6156</td>
            </tr>
            
            <tr>
                <td>Forêt Aléatoire</td>
                <td>1.6138</td>
            </tr>
            
            <tr>
                <td>Gradient Boosting</td>
                <td>1.5979</td>
            </tr>
            
            <tr>
                <td>Extra Trees</td>
                <td>1.6083</td>
            </tr>
            
            <tr>
                <td>XGBoost</td>
                <td>1.5933</td>
            </tr>
            
            <tr>
                <td>LightGBM</td>
                <td>1.5890</td>
            </tr>
            
            <tr>
                <td>MLP</td>
                <td>1.6164</td>
            </tr>
            
            <tr>
                <td>DNN</td>
                <td>1.6254</td>
            </tr>
            
            <tr>
                <td>Ensemble</td>
                <td>1.5945</td>
            </tr>
            
        </table>
    </div>

    <!-- Section 2: Prédictions vs Prix Réels -->
    <div class="section">
        <h2>2. Prédictions vs Prix Réels</h2>
        
            <h3>Prédictions Linéaires vs Réels</h3>
            <p><strong>Prédictions Linéaires</strong></p>
            
                <img src="C:/Users/33766/models/plots\Linear_predictions.png" alt="Linear Predictions">
            
        
            <h3>Prédictions Arbre de Décision vs Réels</h3>
            <p><strong>Prédictions Arbre de Décision</strong></p>
            
                <img src="C:/Users/33766/models/plots\DecisionTree_predictions.png" alt="DecisionTree Predictions">
            
        
            <h3>Prédictions Forêt Aléatoire vs Réels</h3>
            <p><strong>Prédictions Forêt Aléatoire</strong></p>
            
                <img src="C:/Users/33766/models/plots\RandomForest_predictions.png" alt="RandomForest Predictions">
            
        
            <h3>Prédictions Gradient Boosting vs Réels</h3>
            <p><strong>Prédictions Gradient Boosting</strong></p>
            
                <img src="C:/Users/33766/models/plots\GradientBoosting_predictions.png" alt="GradientBoosting Predictions">
            
        
            <h3>Prédictions Extra Trees vs Réels</h3>
            <p><strong>Prédictions Extra Trees</strong></p>
            
                <img src="C:/Users/33766/models/plots\ExtraTrees_predictions.png" alt="ExtraTrees Predictions">
            
        
            <h3>Prédictions XGBoost vs Réels</h3>
            <p><strong>Prédictions XGBoost</strong></p>
            
                <img src="C:/Users/33766/models/plots\XGBoost_predictions.png" alt="XGBoost Predictions">
            
        
            <h3>Prédictions LightGBM vs Réels</h3>
            <p><strong>Prédictions LightGBM</strong></p>
            
                <img src="C:/Users/33766/models/plots\LightGBM_predictions.png" alt="LightGBM Predictions">
            
        
            <h3>Prédictions MLP vs Réels</h3>
            <p><strong>MLP Predictions</strong></p>
            
                <img src="C:/Users/33766/models/plots\MLP_predictions.png" alt="MLP Predictions">
            
        
            <h3>Prédictions DNN vs Réels</h3>
            <p><strong>DNN Predictions</strong></p>
            
                <img src="C:/Users/33766/models/plots\DNN_predictions.png" alt="DNN Predictions">
            
        
            <h3>Prédictions Ensemble vs Réels</h3>
            <p><strong>Prédictions Ensemble</strong></p>
            
                <img src="C:/Users/33766/models/plots\Ensemble_predictions.png" alt="Ensemble Predictions">
            
        
    </div>

    <!-- Section 3: Distribution des Résidus -->
    <div class="section">
        <h2>3. Distribution des Résidus</h2>
        
            <h3>Résidus Linéaires</h3>
            <p><strong>Résidus Linéaires</strong></p>
            
                <img src="C:/Users/33766/models/plots\Linear_residuals.png" alt="Linear Residuals">
            
        
            <h3>Résidus Arbre de Décision</h3>
            <p><strong>Résidus Arbre de Décision</strong></p>
            
                <img src="C:/Users/33766/models/plots\DecisionTree_residuals.png" alt="DecisionTree Residuals">
            
        
            <h3>Résidus Forêt Aléatoire</h3>
            <p><strong>Résidus Forêt Aléatoire</strong></p>
            
                <img src="C:/Users/33766/models/plots\RandomForest_residuals.png" alt="RandomForest Residuals">
            
        
            <h3>Résidus Gradient Boosting</h3>
            <p><strong>Résidus Gradient Boosting</strong></p>
            
                <img src="C:/Users/33766/models/plots\GradientBoosting_residuals.png" alt="GradientBoosting Residuals">
            
        
            <h3>Résidus Extra Trees</h3>
            <p><strong>Résidus Extra Trees</strong></p>
            
                <img src="C:/Users/33766/models/plots\ExtraTrees_residuals.png" alt="ExtraTrees Residuals">
            
        
            <h3>Résidus XGBoost</h3>
            <p><strong>Résidus XGBoost</strong></p>
            
                <img src="C:/Users/33766/models/plots\XGBoost_residuals.png" alt="XGBoost Residuals">
            
        
            <h3>Résidus LightGBM</h3>
            <p><strong>Résidus LightGBM</strong></p>
            
                <img src="C:/Users/33766/models/plots\LightGBM_residuals.png" alt="LightGBM Residuals">
            
        
            <h3>Résidus MLP</h3>
            <p><strong>Résidus MLP</strong></p>
            
                <img src="C:/Users/33766/models/plots\MLP_residuals.png" alt="MLP Residuals">
            
        
            <h3>Résidus DNN</h3>
            <p><strong>Résidus DNN</strong></p>
            
                <img src="C:/Users/33766/models/plots\DNN_residuals.png" alt="DNN Residuals">
            
        
            <h3>Résidus Ensemble</h3>
            <p><strong>Résidus Ensemble</strong></p>
            
                <img src="C:/Users/33766/models/plots\Ensemble_residuals.png" alt="Ensemble Residuals">
            
        
    </div>

    <!-- Section 4: Importances des Caractéristiques -->
    <div class="section">
        <h2>4. Importances des Caractéristiques</h2>
        
            <h3>Importances des Caractéristiques Linéaires</h3>
            
                <img src="C:/Users/33766/models/plots\Linear_feature_importance.png" alt="Linear Feature Importance">
            
        
            <h3>Importances des Caractéristiques Arbre de Décision</h3>
            
                <img src="C:/Users/33766/models/plots\DecisionTree_feature_importance.png" alt="DecisionTree Feature Importance">
            
        
            <h3>Importances des Caractéristiques Forêt Aléatoire</h3>
            
                <img src="C:/Users/33766/models/plots\RandomForest_feature_importance.png" alt="RandomForest Feature Importance">
            
        
            <h3>Importances des Caractéristiques Gradient Boosting</h3>
            
                <img src="C:/Users/33766/models/plots\GradientBoosting_feature_importance.png" alt="GradientBoosting Feature Importance">
            
        
            <h3>Importances des Caractéristiques Extra Trees</h3>
            
                <img src="C:/Users/33766/models/plots\ExtraTrees_feature_importance.png" alt="ExtraTrees Feature Importance">
            
        
            <h3>Importances des Caractéristiques XGBoost</h3>
            
                <img src="C:/Users/33766/models/plots\XGBoost_feature_importance.png" alt="XGBoost Feature Importance">
            
        
            <h3>Importances des Caractéristiques LightGBM</h3>
            
                <img src="C:/Users/33766/models/plots\LightGBM_feature_importance.png" alt="LightGBM Feature Importance">
            
        
            <h3>Importances des Caractéristiques MLP</h3>
            
                <p><em>Aucune importance des caractéristiques disponible pour MLP</em></p>
            
        
            <h3>Importances des Caractéristiques DNN</h3>
            
                <p><em>Aucune importance des caractéristiques disponible pour DNN</em></p>
            
        
            <h3>Importances des Caractéristiques Ensemble</h3>
            
                <p><em>Aucune importance des caractéristiques disponible pour Ensemble</em></p>
            
        
    </div>
</body>
</html>

```
</details>

## Instructions d'Installation et d'Exécution

**Prérequis**

**Python** 3.7 ou supérieur
**pip** installé

## Installation des Dépendances

```
pip install -r requirements.txt
```

## Exécution du Projet

Pour exécuter l'ensemble du pipeline, y compris la préparation des données, l'ingénierie des features, l'entraînement des modèles et la génération des visualisations et du rapport, lancez simplement le script principal :

```
python main.py
```

Ce script exécutera toutes les étapes nécessaires et générera les modèles entraînés, les visualisations et le rapport automatisé.

## Résultats des Modèles

Les performances des différents modèles de régression ont été évaluées à l'aide de la métrique RMSE (Root Mean Squared Error). Un RMSE plus faible indique une meilleure précision des prédictions.

| Modèle                            | RMSE            | 
|-----------------------------------|-----------------|
| Régression Linéaire               | 1.8928          |
| Arbre de Décision                 | 1.6156          | 
| Forêt Aléatoire                   | 1.6138          | 
| Gradient Boosting                 | 1.5979          |
| Extra Trees                       | 1.6083          | 
| XGBoost                           | 1.5933          | 
| LightGBM                          | 1.5890          | 
| MLP (Perceptron Multi-couche)     | 1.6164          | 
| DNN (Réseau de Neurones Profonds) | 1.6254          | 
| Ensemble (Voting Regressor)       | 1.5945          |

### Interprétation des Résultats

**Meilleures Performances :** Les modèles basés sur le boosting, en particulier LightGBM et XGBoost, ont obtenu les meilleurs scores de RMSE, indiquant une excellente capacité à capturer les relations complexes dans les données.

**Approche Ensembliste :** Le modèle Ensemble, combinant plusieurs modèles via un Voting Regressor, a également obtenu un RMSE compétitif, démontrant l'efficacité de l'agrégation des modèles.

**Modèles Linéaires et Simples :** La régression linéaire a le RMSE le plus élevé, suggérant qu'un modèle linéaire simple est moins adapté pour ce type de données.

## Explication des Visualisations

### 1.Graphiques de Prédictions vs Réel

**Objectif :** Visualiser la précision des prédictions de chaque modèle en comparant les valeurs prédites aux valeurs réelles.

**Interprétation :** Un alignement proche de la diagonale indique des prédictions précises.

### 2.Distribution des Résidus

**Objectif :** Illustrer les erreurs de prédiction pour vérifier si elles sont distribuées normalement autour de zéro.

**Interprétation :** Une distribution centrée autour de zéro sans biais indique un modèle bien ajusté.

### 3.Importance des Caractéristiques

**Objectif :** Identifier les variables qui influencent le plus les prédictions du modèle.

**Interprétation :** Les variables avec une importance élevée sont des facteurs clés dans la variation des prix des appartements.

## Conclusion

Ce projet démontre l'efficacité des techniques de régression avancées pour prédire les prix des appartements à Paris en se basant sur divers facteurs géographiques et caractéristiques des propriétés. Les modèles basés sur le boosting, tels que LightGBM et XGBoost, se sont avérés être les plus performants, obtenant les meilleurs scores de RMSE. L'approche ensembliste, qui combine plusieurs modèles, a également montré de bons résultats, renforçant l'idée que l'agrégation de modèles peut améliorer la précision des prédictions.

Grâce aux visualisations et aux rapports générés, les utilisateurs peuvent non seulement obtenir des prédictions précises, mais aussi mieux comprendre les facteurs influençant les prix immobiliers. Cela permet de prendre des décisions informées, que ce soit pour l'achat, la vente ou l'évaluation des biens immobiliers à Paris.
