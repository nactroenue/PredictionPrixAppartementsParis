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
