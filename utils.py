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
