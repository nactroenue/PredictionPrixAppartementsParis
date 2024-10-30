# config.py

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
