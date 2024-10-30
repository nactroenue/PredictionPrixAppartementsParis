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
