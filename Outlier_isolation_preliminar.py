
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           roc_auc_score, precision_recall_curve, roc_curve)
from sklearn.preprocessing import StandardScaler
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('default')
sns.set_palette("husl")

class AdvancedOutlierAnalysis:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.data = None
        self.predictors = None
        self.model = None
        self.outlier_scores = None
        self.results_dir = self.file_path.parent / "outlier_analysis_results"
        self.results_dir.mkdir(exist_ok=True)
        print(f"📂 Directorio creado: {self.results_dir}")
        
    def load_and_preprocess_data(self):
        """Cargar y preprocesar los datos"""
        print("🔄 Cargando y preprocesando datos...")
        
        # Cargar datos
        self.data = pd.read_csv(self.file_path)
        print(f"   Datos cargados: {self.data.shape}")
        
        # Reorganizar columnas
        current_columns = self.data.columns.tolist()
        desired_names = [
            "TF_CENTRO_POBLADO_NOMBRE",
            "TF_DEPARTAMENTO_NOMBRE",
            "TF_MUNICIPIO_NOMBRE",
        ]
        
        existing_desired = [col for col in desired_names if col in current_columns]
        new_columns = existing_desired + [col for col in current_columns if col not in existing_desired]
        self.data = self.data[new_columns]
        
        # Convertir formato de listas
        print("   Procesando formato de listas...")
        processed_count = 0
        for col in self.data.columns[9:]:
            try:
                sample = self.data[col].dropna().head(5)
                needs_processing = any(isinstance(x, str) and '[' in str(x) for x in sample)
                
                if needs_processing:
                    self.data[col] = self.data[col].apply(
                        lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and '[' in x else x
                    )
                    processed_count += 1
            except Exception as e:
                continue
        
        print(f"   Columnas procesadas: {processed_count}")
        
        # Seleccionar predictores
        self.predictors = self.data.iloc[:, 32:].copy()
        print(f"   Predictores seleccionados: {self.predictors.shape}")
        
        # Limpiar datos
        self._clean_predictors()
        
    def _clean_predictors(self):
        """Limpiar datos predictores"""
        print("   🧹 Limpiando datos...")
        
        original_shape = self.predictors.shape
        
        # Convertir a numérico
        for col in self.predictors.columns:
            if not pd.api.types.is_numeric_dtype(self.predictors[col]):
                self.predictors[col] = pd.to_numeric(self.predictors[col], errors='coerce')
        
        # Filtrar columnas válidas
        valid_cols = []
        for col in self.predictors.columns:
            valid_ratio = self.predictors[col].notna().sum() / len(self.predictors)
            if valid_ratio >= 0.5:
                valid_cols.append(col)
        
        self.predictors = self.predictors[valid_cols]
        
        # Rellenar valores faltantes
        self.predictors = self.predictors.fillna(self.predictors.median())
        
        # Manejar infinitos
        self.predictors = self.predictors.replace([np.inf, -np.inf], np.nan)
        self.predictors = self.predictors.fillna(self.predictors.median())
        
        print(f"   Limpieza: {original_shape} → {self.predictors.shape}")
        
    def optimize_hyperparameters(self, predictors_scaled):
        """Optimizar hiperparámetros usando GridSearchCV"""
        from sklearn.model_selection import GridSearchCV
        
        print("🔍 Optimizando hiperparámetros...")
        
        # Definir grid de parámetros (SIN contamination - se mantiene en 'auto')
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_samples': ['auto', 0.7, 1.0],
            'max_features': [0.7, 1.0]
        }
        
        # Modelo base CON contamination='auto' FIJO
        base_model = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
        
        # Función de puntuación personalizada para Isolation Forest
        def isolation_scorer(estimator, X):
            scores = estimator.decision_function(X)
            return np.mean(scores)  # Maximizar puntuación media
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring=isolation_scorer,
            n_jobs=-1,
            verbose=0
        )
        
        # Entrenar con submuestra para velocidad
        sample_size = min(5000, len(predictors_scaled))
        sample_indices = np.random.choice(len(predictors_scaled), sample_size, replace=False)
        grid_search.fit(predictors_scaled[sample_indices])
        
        self.best_params = grid_search.best_params_
        # Añadir contamination='auto' a los mejores parámetros para mostrar
        self.best_params['contamination'] = 'auto'
        print(f"   ✅ Mejores parámetros: {self.best_params}")
        
        return self.best_params
        
    def train_model(self, optimize_hyperparams=False):
        """Entrenar el modelo Isolation Forest"""
        print("🤖 Entrenando modelo Isolation Forest...")
        
        # Escalar datos
        scaler = StandardScaler()
        predictors_scaled = scaler.fit_transform(self.predictors)
        
        # Optimizar hiperparámetros si se solicita
        if optimize_hyperparams:
            best_params = self.optimize_hyperparameters(predictors_scaled)
            # Crear modelo con mejores parámetros pero manteniendo contamination='auto'
            self.model = IsolationForest(
                n_estimators=best_params['n_estimators'],
                max_samples=best_params['max_samples'],
                max_features=best_params['max_features'],
                contamination='auto',  # SIEMPRE automático
                random_state=42,
                n_jobs=-1
            )
        else:
            # Usar parámetros por defecto mejorados
            self.model = IsolationForest(
                n_estimators=200,
                contamination='auto',  # SIEMPRE automático
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(predictors_scaled)
        
        # Obtener puntuaciones y predicciones
        self.outlier_scores = self.model.decision_function(predictors_scaled)
        outliers = self.model.predict(predictors_scaled)
        
        # Añadir resultados al DataFrame
        self.data["Outlier_Score"] = self.outlier_scores
        self.data["Outlier"] = (outliers == -1).astype(int)
        
        outlier_count = self.data['Outlier'].sum()
        outlier_pct = outlier_count / len(self.data) * 100
        
        print(f"   ✅ Modelo entrenado con contamination='auto'")
        print(f"   📊 Outliers: {outlier_count:,} de {len(self.data):,} ({outlier_pct:.2f}%)")
        
    def calculate_advanced_metrics(self):
        """Calcular métricas avanzadas"""
        print("📊 Calculando métricas avanzadas...")
        
        if "ATP" not in self.data.columns:
            print("   ⚠️ Columna 'ATP' no encontrada. Métricas limitadas.")
            self.metrics = None
            return None
        
        ground_truth = self.data["ATP"]
        predictions = self.data["Outlier"]
        scores = -self.outlier_scores
        
        # Métricas básicas
        accuracy = accuracy_score(ground_truth, predictions)
        conf_matrix = confusion_matrix(ground_truth, predictions)
        class_report = classification_report(ground_truth, predictions, output_dict=True)
        
        # Métricas avanzadas
        try:
            auc_score = roc_auc_score(ground_truth, scores)
            fpr, tpr, _ = roc_curve(ground_truth, scores)
            precision, recall, _ = precision_recall_curve(ground_truth, scores)
        except Exception as e:
            print(f"   ⚠️ Error calculando AUC: {e}")
            auc_score = None
            fpr = tpr = precision = recall = None
        
        self.metrics = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'auc_score': auc_score,
            'classification_report': class_report,
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision, recall)
        }
        
        print(f"   ✅ Métricas calculadas - Accuracy: {accuracy:.4f}")
        if auc_score:
            print(f"   ✅ AUC-ROC: {auc_score:.4f}")
        
        return self.metrics
    
    def create_comprehensive_plots(self):
        """Crear gráficos comprehensivos"""
        print("📈 Creando visualizaciones...")
        
        # Configurar subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Distribución de puntuaciones de anomalía
        ax1 = plt.subplot(3, 3, 1)
        plt.hist(self.outlier_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.percentile(self.outlier_scores, 10), color='red', linestyle='--', 
                   label=f'Percentil 10: {np.percentile(self.outlier_scores, 10):.3f}')
        plt.title('Distribución de Puntuaciones de Anomalía', fontsize=14, fontweight='bold')
        plt.xlabel('Puntuación de Anomalía')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Outliers por departamento
        ax2 = plt.subplot(3, 3, 2)
        if 'TF_DEPARTAMENTO_NOMBRE' in self.data.columns:
            dept_stats = self.data.groupby('TF_DEPARTAMENTO_NOMBRE')['Outlier'].agg(['count', 'sum']).reset_index()
            dept_stats['percentage'] = (dept_stats['sum'] / dept_stats['count']) * 100
            top_depts = dept_stats.nlargest(10, 'percentage')
            
            bars = plt.barh(range(len(top_depts)), top_depts['percentage'], color='coral')
            plt.yticks(range(len(top_depts)), top_depts['TF_DEPARTAMENTO_NOMBRE'], fontsize=10)
            plt.title('Top 10 Departamentos con Mayor % de Outliers', fontsize=14, fontweight='bold')
            plt.xlabel('Porcentaje de Outliers')
            
            # Añadir valores en las barras
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f}%', ha='left', va='center', fontsize=9)
        
        # 3. Matriz de confusión
        ax3 = plt.subplot(3, 3, 3)
        if hasattr(self, 'metrics') and self.metrics:
            sns.heatmap(self.metrics['confusion_matrix'], annot=True, fmt='d', 
                       cmap='Blues', square=True, cbar_kws={'shrink': 0.8})
            plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
            plt.xlabel('Predicción')
            plt.ylabel('Valor Real')
        
        # 4. Comparación de distribuciones
        ax4 = plt.subplot(3, 3, 4)
        if len(self.predictors.columns) > 0:
            first_col = self.predictors.columns[0]
            normal_data = self.predictors[self.data['Outlier'] == 0][first_col]
            outlier_data = self.predictors[self.data['Outlier'] == 1][first_col]
            
            plt.hist(normal_data, bins=30, alpha=0.7, label='Normal', color='green', density=True)
            plt.hist(outlier_data, bins=30, alpha=0.7, label='Outlier', color='red', density=True)
            plt.title(f'Distribución: {first_col}', fontsize=14, fontweight='bold')
            plt.xlabel(first_col)
            plt.ylabel('Densidad')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. Curva ROC
        ax5 = plt.subplot(3, 3, 5)
        if hasattr(self, 'metrics') and self.metrics and self.metrics['roc_curve'][0] is not None:
            fpr, tpr = self.metrics['roc_curve']
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {self.metrics["auc_score"]:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
        
        # 6. Curva Precision-Recall
        ax6 = plt.subplot(3, 3, 6)
        if hasattr(self, 'metrics') and self.metrics and self.metrics['pr_curve'][0] is not None:
            precision, recall = self.metrics['pr_curve']
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Curva Precision-Recall', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
        
        # 7. Distribución por municipios (top 15)
        ax7 = plt.subplot(3, 3, 7)
        if 'TF_MUNICIPIO_NOMBRE' in self.data.columns:
            mun_stats = self.data.groupby('TF_MUNICIPIO_NOMBRE')['Outlier'].agg(['count', 'sum']).reset_index()
            mun_stats = mun_stats[mun_stats['count'] >= 10]
            mun_stats['percentage'] = (mun_stats['sum'] / mun_stats['count']) * 100
            top_muns = mun_stats.nlargest(15, 'percentage')
            
            if len(top_muns) > 0:
                plt.barh(range(len(top_muns)), top_muns['percentage'], color='lightcoral')
                plt.yticks(range(len(top_muns)), 
                          [name[:20] + '...' if len(name) > 20 else name for name in top_muns['TF_MUNICIPIO_NOMBRE']], 
                          fontsize=8)
                plt.title('Top 15 Municipios con Mayor % de Outliers', fontsize=14, fontweight='bold')
                plt.xlabel('Porcentaje de Outliers')
        
        # 8. Heatmap de correlaciones (TODAS las variables disponibles)
        ax8 = plt.subplot(3, 3, 8)
        if len(self.predictors.columns) >= 2:
            # Usar todas las variables o las primeras 20 si hay muchas
            max_vars = min(20, len(self.predictors.columns))
            corr_data = self.predictors.iloc[:, :max_vars].corr()
            
            # Ajustar tamaño de fuente según número de variables
            if max_vars <= 10:
                annot_size = 8
                tick_size = 8
            elif max_vars <= 15:
                annot_size = 6
                tick_size = 7
            else:
                annot_size = 4
                tick_size = 6
            
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8},
                       annot_kws={'size': annot_size})
            plt.title(f'Correlaciones ({max_vars} variables)', fontsize=14, fontweight='bold')
            plt.xticks(fontsize=tick_size, rotation=45)
            plt.yticks(fontsize=tick_size, rotation=0)
        
        # 9. Boxplot de puntuaciones por clase
        ax9 = plt.subplot(3, 3, 9)
        outlier_labels = ['Normal', 'Outlier']
        score_data = [self.outlier_scores[self.data['Outlier'] == 0],
                     self.outlier_scores[self.data['Outlier'] == 1]]
        
        box_plot = plt.boxplot(score_data, labels=outlier_labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        plt.title('Distribución de Puntuaciones por Clase', fontsize=14, fontweight='bold')
        plt.ylabel('Puntuación de Anomalía')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráficos
        plot_path = self.results_dir / "comprehensive_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Gráficos guardados en: {plot_path}")
        plt.show()
        
    def generate_detailed_report(self):
        """Generar reporte detallado"""
        print("📋 Generando reporte detallado...")
        
        report_path = self.results_dir / "detailed_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE DETALLADO DE ANÁLISIS DE OUTLIERS - AGROSAVIA\n")
            f.write("="*80 + "\n\n")
            f.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Archivo analizado: {self.file_path.name}\n\n")
            
            # Estadísticas generales
            f.write("ESTADÍSTICAS GENERALES:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total de registros: {len(self.data):,}\n")
            f.write(f"Variables predictoras utilizadas: {len(self.predictors.columns)}\n")
            f.write(f"Outliers detectados: {self.data['Outlier'].sum():,}\n")
            f.write(f"Porcentaje de outliers: {self.data['Outlier'].sum()/len(self.data)*100:.2f}%\n\n")
            
            # Estadísticas de puntuaciones
            f.write("ESTADÍSTICAS DE PUNTUACIONES DE ANOMALÍA:\n")
            f.write("-" * 45 + "\n")
            f.write(f"Media: {np.mean(self.outlier_scores):.4f}\n")
            f.write(f"Mediana: {np.median(self.outlier_scores):.4f}\n")
            f.write(f"Desviación estándar: {np.std(self.outlier_scores):.4f}\n")
            f.write(f"Mínimo: {np.min(self.outlier_scores):.4f}\n")
            f.write(f"Máximo: {np.max(self.outlier_scores):.4f}\n")
            f.write(f"Percentil 10: {np.percentile(self.outlier_scores, 10):.4f}\n")
            f.write(f"Percentil 90: {np.percentile(self.outlier_scores, 90):.4f}\n\n")
            
            # NUEVA SECCIÓN: Parámetros del modelo
            f.write("PARÁMETROS DEL MODELO ISOLATION FOREST:\n")
            f.write("-" * 40 + "\n")
            if self.model:
                f.write(f"n_estimators: {self.model.n_estimators}\n")
                f.write(f"contamination: {self.model.contamination}\n")
                f.write(f"max_samples: {self.model.max_samples}\n")
                f.write(f"max_features: {self.model.max_features}\n")
                f.write(f"bootstrap: {self.model.bootstrap}\n")
                f.write(f"random_state: {self.model.random_state}\n")
                f.write(f"n_jobs: {self.model.n_jobs}\n")
                f.write(f"warm_start: {self.model.warm_start}\n")
                
                # Si hay parámetros optimizados, mostrarlos
                if hasattr(self, 'best_params'):
                    f.write(f"\nPARÁMETROS OPTIMIZADOS:\n")
                    f.write("-" * 22 + "\n")
                    for param, value in self.best_params.items():
                        f.write(f"{param}: {value}\n")
                    f.write("Nota: Estos son los parámetros encontrados mediante GridSearchCV\n")
                else:
                    f.write(f"\nTipo de configuración: Parámetros por defecto (no optimizados)\n")
            f.write("\n")
            
            # Métricas del modelo
            if hasattr(self, 'metrics') and self.metrics:
                f.write("MÉTRICAS DEL MODELO:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Accuracy: {self.metrics['accuracy']:.4f}\n")
                if self.metrics['auc_score']:
                    f.write(f"AUC-ROC: {self.metrics['auc_score']:.4f}\n")
                
                # Métricas detalladas por clase
                cr = self.metrics['classification_report']
                f.write(f"\nPrecision (Clase 0): {cr['0']['precision']:.4f}\n")
                f.write(f"Recall (Clase 0): {cr['0']['recall']:.4f}\n")
                f.write(f"F1-Score (Clase 0): {cr['0']['f1-score']:.4f}\n")
                f.write(f"Precision (Clase 1): {cr['1']['precision']:.4f}\n")
                f.write(f"Recall (Clase 1): {cr['1']['recall']:.4f}\n")
                f.write(f"F1-Score (Clase 1): {cr['1']['f1-score']:.4f}\n\n")
                
                f.write("MATRIZ DE CONFUSIÓN:\n")
                f.write("-" * 20 + "\n")
                f.write(f"{self.metrics['confusion_matrix']}\n\n")
            
            # Top departamentos con outliers
            if 'TF_DEPARTAMENTO_NOMBRE' in self.data.columns:
                dept_stats = self.data.groupby('TF_DEPARTAMENTO_NOMBRE')['Outlier'].agg(['count', 'sum']).reset_index()
                dept_stats['percentage'] = (dept_stats['sum'] / dept_stats['count']) * 100
                top_depts = dept_stats.nlargest(10, 'percentage')
                
                f.write("TOP 10 DEPARTAMENTOS CON MAYOR % DE OUTLIERS:\n")
                f.write("-" * 50 + "\n")
                for _, row in top_depts.iterrows():
                    f.write(f"{row['TF_DEPARTAMENTO_NOMBRE']}: {row['sum']}/{row['count']} ({row['percentage']:.1f}%)\n")
                f.write("\n")
            
            # Variables utilizadas
            f.write("VARIABLES PREDICTORAS UTILIZADAS:\n")
            f.write("-" * 35 + "\n")
            for i, col in enumerate(self.predictors.columns, 1):
                f.write(f"{i:2d}. {col}\n")
            
            # Resumen de correlaciones
            f.write(f"\nRESUMEN DE CORRELACIONES:\n")
            f.write("-" * 25 + "\n")
            if len(self.predictors.columns) >= 2:
                corr_matrix = self.predictors.corr()
                
                # Encontrar correlaciones más altas (excluyendo diagonal)
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:  # Solo correlaciones significativas
                            high_corr_pairs.append((
                                corr_matrix.columns[i], 
                                corr_matrix.columns[j], 
                                corr_val
                            ))
                
                if high_corr_pairs:
                    f.write("Correlaciones altas (|r| > 0.5):\n")
                    # Ordenar por valor absoluto de correlación
                    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    for var1, var2, corr in high_corr_pairs[:10]:  # Top 10
                        f.write(f"  {var1} - {var2}: {corr:.3f}\n")
                else:
                    f.write("No se encontraron correlaciones altas (|r| > 0.5)\n")
                
                # Estadísticas generales de correlación
                all_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        all_corrs.append(abs(corr_matrix.iloc[i, j]))
                
                if all_corrs:
                    f.write(f"\nEstadísticas de correlación:\n")
                    f.write(f"  Promedio: {np.mean(all_corrs):.3f}\n")
                    f.write(f"  Máximo: {np.max(all_corrs):.3f}\n")
                    f.write(f"  Mínimo: {np.min(all_corrs):.3f}\n")
        
        print(f"   📋 Reporte guardado en: {report_path}")
        return report_path
    
    def export_results(self):
        """Exportar todos los resultados"""
        print("💾 Exportando resultados...")
        
        # CSV con resultados
        csv_path = self.results_dir / "outlier_results.csv"
        self.data.to_csv(csv_path, index=False)
        
        # CSV solo con outliers
        outliers_only = self.data[self.data['Outlier'] == 1].copy()
        outliers_csv_path = self.results_dir / "outliers_only.csv"
        outliers_only.to_csv(outliers_csv_path, index=False)
        
        print(f"   📁 Resultados completos: {csv_path}")
        print(f"   🚨 Solo outliers: {outliers_csv_path}")
        
        return csv_path, outliers_csv_path
    
    def run_complete_analysis(self, optimize_hyperparams=False):
        """Ejecutar análisis completo"""
        print("🚀 INICIANDO ANÁLISIS COMPLETO DE OUTLIERS")
        print("="*60)
        
        if optimize_hyperparams:
            print("🔧 Modo: CON optimización de hiperparámetros")
        else:
            print("⚡ Modo: SIN optimización (más rápido)")
            
        try:
            # 1. Cargar y preprocesar
            self.load_and_preprocess_data()
            
            # 2. Entrenar modelo
            self.train_model(optimize_hyperparams=optimize_hyperparams)
            
            # 3. Calcular métricas
            self.calculate_advanced_metrics()
            
            # 4. Crear gráficos
            self.create_comprehensive_plots()
            
            # 5. Generar reporte
            report_path = self.generate_detailed_report()
            
            # 6. Exportar resultados
            csv_path, outliers_path = self.export_results()
            
            print("\n" + "="*60)
            print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
            print("="*60)
            print(f"📂 Directorio de resultados: {self.results_dir}")
            print(f"📊 Outliers detectados: {self.data['Outlier'].sum():,} de {len(self.data):,}")
            print(f"📈 Porcentaje: {self.data['Outlier'].sum()/len(self.data)*100:.2f}%")
            
            if hasattr(self, 'best_params'):
                print(f"🔧 Hiperparámetros óptimos: {self.best_params}")
            
            if hasattr(self, 'metrics') and self.metrics:
                print(f"🎯 Accuracy: {self.metrics['accuracy']:.4f}")
                if self.metrics['auc_score']:
                    print(f"📊 AUC-ROC: {self.metrics['auc_score']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error durante el análisis: {e}")
            import traceback
            traceback.print_exc()
            return False

# Ejecutar análisis
if __name__ == "__main__":
    file_path = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\grupo 4_rows_45311_value_columns_18.csv"
    
    # Crear instancia
    analyzer = AdvancedOutlierAnalysis(file_path)
    
    # OPCIÓN 1: Ejecutar SIN optimización (más rápido)
    success = analyzer.run_complete_analysis(optimize_hyperparams=False)
    
    # OPCIÓN 2: Ejecutar CON optimización de hiperparámetros (más lento pero mejor)
    # success = analyzer.run_complete_analysis(optimize_hyperparams=True)
    
    if success:
        print("\n🎉 ¡Análisis completado! Revisa la carpeta 'outlier_analysis_results'")
    else:
        print("\n💔 El análisis no se pudo completar. Revisa los errores anteriores.")