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
