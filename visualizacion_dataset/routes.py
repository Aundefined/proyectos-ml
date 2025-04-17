from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import time
import os
import json
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import traceback
import logging
# Configurar el logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Crear un Blueprint para la visualización de datasets
from . import visualizacion_dataset_bp

# Clase personalizada para manejar la serialización de NaN
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)

def generar_grafico_base64(plt_figure):
    """Convierte un gráfico de matplotlib a una imagen base64 para mostrar en HTML"""
    img = io.BytesIO()
    plt_figure.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(plt_figure)
    return plot_url

def sugerir_preprocesamiento(df, columna):
    """Sugiere técnicas de preprocesamiento para una columna específica"""
    # Verificar el tipo de datos
    dtype = df[columna].dtype
    n_unique = df[columna].nunique()
    n_missing = df[columna].isna().sum()
    total = len(df)
    
    sugerencias = []
    
    # Verificar valores nulos
    if n_missing > 0:
        porcentaje_nulos = (n_missing / total) * 100
        if porcentaje_nulos > 30:
            sugerencias.append(f"Alta cantidad de nulos ({porcentaje_nulos:.2f}%). Considere eliminar esta columna o imputar.")
        else:
            sugerencias.append(f"Contiene {n_missing} valores nulos ({porcentaje_nulos:.2f}%). Se recomienda imputar.")
    
    # Sugerencias específicas por tipo de dato
    if pd.api.types.is_numeric_dtype(dtype):
        # Para columnas numéricas
        if df[columna].min() >= 0 and df[columna].max() <= 1:
            sugerencias.append("Ya parece estar normalizada (0-1).")
        else:
            # Verificar distribución para normalización
            skew = df[columna].skew()
            if abs(skew) > 1:
                sugerencias.append(f"Distribución sesgada (skew={skew:.2f}). Considere transformación logarítmica o Box-Cox.")
            else:
                sugerencias.append("Se recomienda estandarización (StandardScaler) o Min-Max scaling.")
                
        # Verificar si podría ser categórica codificada como numérica
        if n_unique < 10 and n_unique / total < 0.05:
            sugerencias.append("Posible variable categórica. Considere one-hot encoding.")
            
    elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
        # Para columnas categóricas/texto
        if n_unique == 2:
            sugerencias.append("Variable binaria. Se recomienda codificación con LabelEncoder.")
        elif n_unique <= 10:
            sugerencias.append("Se recomienda One-Hot Encoding.")
        else:
            sugerencias.append(f"Alta cardinalidad ({n_unique} valores únicos). Considere Target Encoding o agrupar categorías menos frecuentes.")
            
    elif pd.api.types.is_datetime64_dtype(dtype):
        # Para fechas
        sugerencias.append("Extraiga características como año, mes, día, día de la semana, etc.")
    
    return sugerencias

def calcular_tiempo_estimado(n_rows, n_cols):
    """Calcula un tiempo estimado de procesamiento basado en el tamaño del dataset"""
    # Fórmula simple: los factores pueden ajustarse según benchmarks reales
    base_time = 0.5  # tiempo base en segundos
    row_factor = 0.0001  # tiempo adicional por fila
    col_factor = 0.01  # tiempo adicional por columna
    
    estimated_time = base_time + (n_rows * row_factor) + (n_cols * col_factor)
    
    # Convertir a minutos si es mayor a 60 segundos
    if estimated_time > 60:
        return f"{estimated_time/60:.2f} minutos"
    else:
        return f"{estimated_time:.2f} segundos"

@visualizacion_dataset_bp.route('/', methods=['GET', 'POST'])
def index():
    """Página para la visualización de datasets"""
    if request.method == 'GET':
        return render_template('visualizacion_dataset.html')
    
    # Procesar el archivo subido
    if 'file' not in request.files:
        return jsonify({'error': 'No se ha subido ningún archivo'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se ha seleccionado ningún archivo'}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'El archivo debe ser un CSV'}), 400
    
    # Obtener el índice de la etiqueta si se especificó
    label_index = request.form.get('labelIndex', '')
    
    # Obtener el límite de filas si se especificó
    max_rows = request.form.get('maxRows', '')
    max_rows = int(max_rows) if max_rows and max_rows.isdigit() else None
    
    try:
        # Guardar temporalmente el archivo
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        
        # Leer el CSV, limitando las filas si se especificó max_rows
        try:
            if max_rows:
                df = pd.read_csv(temp_path, encoding='utf-8', nrows=max_rows)
            else:
                df = pd.read_csv(temp_path, encoding='utf-8')
        except UnicodeDecodeError:
            if max_rows:
                df = pd.read_csv(temp_path, encoding='latin1', nrows=max_rows)
            else:
                df = pd.read_csv(temp_path, encoding='latin1')

        
       # Determinar la columna de etiqueta
        label_column_name = request.form.get('labelColumn', '').strip()

        if label_column_name:
            # Buscar la columna sin importar mayúsculas/minúsculas
            column_found = False
            for col in df.columns:
                if col.lower() == label_column_name.lower():
                    label_column = col  # Usar el nombre exacto de la columna como está en el DataFrame
                    column_found = True
                    break
    
            if not column_found:
                return jsonify({'error': f'No se encontró la columna "{label_column_name}" en el dataset'}), 400
        else:
         # Por defecto, usar la última columna como etiqueta
            label_column = df.columns[-1]
        
        # Información básica del dataset
        info_buffer = io.StringIO()
        df.info(buf=info_buffer)
        info_str = info_buffer.getvalue()
        
        # Descripción estadística
        describe_html = df.describe(include='all').to_html(classes='table table-striped table-sm')
        
        # Tipos de datos por columna
        dtypes = {col: str(df[col].dtype) for col in df.columns}
        
        # Preparar y limpiar los datos para el modelo
        X = df.drop(label_column, axis=1)
        y = df[label_column]
        
        # Preprocesar datos categóricos y valores faltantes para el modelo
        X_processed = X.copy()
        
        # Manejar columnas categóricas
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            # Llenar valores nulos con un marcador especial
            X_processed[col] = X_processed[col].fillna('Missing')
            # Codificar categorías
            X_processed[col] = LabelEncoder().fit_transform(X_processed[col].astype(str))
        
        # Manejar valores numéricos faltantes
        numeric_cols = X.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        
        # Determinar si es un problema de clasificación o regresión
        is_classification = pd.api.types.is_categorical_dtype(y) or y.nunique() < 10
        
        if is_classification:
            # Para clasificación, manejar valores nulos en y
            y_train = y_train.fillna('Missing')
            y_test = y_test.fillna('Missing')
            
            # Codificar etiquetas
            y_encoder = LabelEncoder()
            y_train = y_encoder.fit_transform(y_train.astype(str))
            y_test = y_encoder.transform(y_test.astype(str))
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            # Para regresión, manejar valores nulos en y
            y_train = y_train.fillna(y_train.median())
            y_test = y_test.fillna(y_test.median())
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_test)
        if is_classification:
            score = accuracy_score(y_test, y_pred)
            metric_name = "Accuracy"
        else:
            score = r2_score(y_test, y_pred)
            metric_name = "R²"
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Graficar feature importance
        fig_importance, ax_importance = plt.subplots(figsize=(10, 8))
        feature_importance.head(10).plot(x='feature', y='importance', kind='barh', ax=ax_importance)
        ax_importance.set_title('Importancia de Características (Top 10)')
        ax_importance.set_xlabel('Importancia')
        plt.tight_layout()
        plot_importance = generar_grafico_base64(fig_importance)
        
        # Matriz de correlación
        # Usar solo columnas numéricas para la correlación
        numeric_df = X_processed.select_dtypes(include=['number'])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax_corr)
            ax_corr.set_title('Matriz de Correlación')
            plt.tight_layout()
            plot_corr = generar_grafico_base64(fig_corr)
        else:
            plot_corr = ""
        
        # Gráficos de dispersión para las características más importantes
        scatter_plots = []
        top_features = feature_importance.head(3)['feature'].values
        
        if len(top_features) >= 2 and all(f in numeric_df.columns for f in top_features[:2]):
            fig, ax = plt.subplots(figsize=(10, 6))
            if is_classification:
                scatter = ax.scatter(X[top_features[0]], X[top_features[1]], c=y.astype('category').cat.codes, cmap='viridis', alpha=0.6)
                ax.set_title(f'Dispersión: {top_features[0]} vs {top_features[1]}, coloreado por {label_column}')
                plt.colorbar(scatter, ax=ax, label=label_column)
            else:
                # Asegurar que y sea numérico para el coloreo
                y_numeric = pd.to_numeric(y, errors='coerce')
                scatter = ax.scatter(X[top_features[0]], X[top_features[1]], c=y_numeric, cmap='viridis', alpha=0.6)
                ax.set_title(f'Dispersión: {top_features[0]} vs {top_features[1]}, coloreado por {label_column}')
                plt.colorbar(scatter, ax=ax, label=label_column)
                
            ax.set_xlabel(top_features[0])
            ax.set_ylabel(top_features[1])
            plt.tight_layout()
            plot_scatter = generar_grafico_base64(fig)
            scatter_plots.append({
                'title': f'{top_features[0]} vs {top_features[1]}',
                'image': plot_scatter
            })
        
        # Histogramas para las características principales
        histograms = []
        for feature in top_features:
            if feature in numeric_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                X[feature].hist(bins=30, ax=ax)
                ax.set_title(f'Distribución de {feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frecuencia')
                plt.tight_layout()
                plot_hist = generar_grafico_base64(fig)
                histograms.append({
                    'title': f'Distribución de {feature}',
                    'image': plot_hist
                })
        
        # Sugerencias de preprocesamiento
        preprocessing_suggestions = {}
        for col in df.columns:
            preprocessing_suggestions[col] = sugerir_preprocesamiento(df, col)
        
        # Preparar los datos para la tabla paginable
        # Convertir NaNs a None para serialización JSON
        table_data = []
        for _, row in df.head(1000).iterrows():  # Limitamos a 1000 filas para la visualización en tabla
            row_dict = {}
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    row_dict[col] = None
                else:
                    row_dict[col] = value
            table_data.append(row_dict)
        
        # Obtener estadísticas de valores faltantes
        missing_values = df.isnull().sum().to_dict()
        missing_percent = (df.isnull().sum() / len(df) * 100).to_dict()
        missing_stats = {}
        for col in missing_values:
            missing_stats[col] = {
                'count': int(missing_values[col]),
                'percent': float(missing_percent[col])
            }
        
        # Eliminar el archivo temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Preparar el resultado en formato JSON
        columns_list = df.columns.tolist()
        dtypes_dict = {col: str(df[col].dtype) for col in columns_list}
        
        # Convertir DataFrame de importancia a lista de diccionarios
        feature_importance_list = []
        for _, row in feature_importance.iterrows():
            feature_importance_list.append({
                'feature': row['feature'],
                'importance': float(row['importance'])
            })
        
        # Añadir información sobre el límite de filas aplicado
        result = {
            'fileName': file.filename,
            'rowCount': len(df),
            'totalRowsInFile': 'Desconocido' if max_rows is None else 'Al menos ' + str(max_rows),
            'rowsLimited': max_rows is not None,
            'maxRowsApplied': max_rows,
            'columnCount': len(df.columns),
            'labelColumn': label_column,
            'columns': columns_list,
            'dtypes': dtypes_dict,
            'info': info_str,
            'describe': describe_html,
            'tableData': table_data,
            'featureImportance': feature_importance_list,
            'plotImportance': plot_importance,
            'plotCorrelation': plot_corr,
            'scatterPlots': scatter_plots,
            'histograms': histograms,
            'preprocessingSuggestions': preprocessing_suggestions,
            'missingStats': missing_stats,
            'modelMetric': {
                'name': metric_name,
                'value': float(score)
            },
            'isClassification': is_classification
        }
        
        # Usar dumps con el encoder personalizado y luego loads para serializar bien
        json_str = json.dumps(result, cls=NpEncoder)
        return json.loads(json_str)
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
    
# Añadir esta nueva ruta al archivo visualizacion_dataset.py

@visualizacion_dataset_bp.route('/preview/', methods=['POST'])
def preview_csv():
    """Endpoint para obtener una vista previa del CSV cargado"""
    if 'file' not in request.files:
        return jsonify({'error': 'No se ha subido ningún archivo'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se ha seleccionado ningún archivo'}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'El archivo debe ser un CSV'}), 400
    
    try:
        # Guardar temporalmente el archivo
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        # Leer el CSV (primeras 100 filas para la vista previa)
        try:
            df = pd.read_csv(temp_path, encoding='utf-8', nrows=100)
        except UnicodeDecodeError:
            df = pd.read_csv(temp_path, encoding='latin1', nrows=100)
        
        # Preparar los datos para la tabla paginable
        table_data = []
        for _, row in df.iterrows():
            row_dict = {}
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    row_dict[col] = None
                else:
                    row_dict[col] = value
            table_data.append(row_dict)
        
        # Obtener información básica
        columns_list = df.columns.tolist()
        dtypes_dict = {col: str(df[col].dtype) for col in columns_list}
        
        # Eliminar el archivo temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Preparar el resultado en formato JSON (solo datos básicos)
        result = {
            'fileName': file.filename,
            'rowCount': len(df),
            'columnCount': len(df.columns),
            'columns': columns_list,
            'dtypes': dtypes_dict,
            'tableData': table_data
        }
        
        # Usar dumps con el encoder personalizado y luego loads para serializar bien
        json_str = json.dumps(result, cls=NpEncoder)
        return json.loads(json_str)
        
    except Exception as e:
        logger.error(f"Error previewing CSV: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500