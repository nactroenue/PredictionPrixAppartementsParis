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
