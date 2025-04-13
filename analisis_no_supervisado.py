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
import logging
from scipy import stats
import tempfile
import traceback

# Configurar el logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear un Blueprint para el análisis no supervisado
analisis_no_supervisado_bp = Blueprint('analisis_no_supervisado', __name__, url_prefix='/analisis-no-supervisado')

# Clase para manejar la serialización de NaN y otros tipos numpy
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
            sugerencias.append(f"Contiene {n_missing} valores nulos ({porcentaje_nulos:.2f}%).")
            
            # Sugerir método de imputación según tipo de datos
            if pd.api.types.is_numeric_dtype(dtype):
                sugerencias.append("Para columnas numéricas, imputar con media, mediana o KNN para preservar relaciones.")
            else:
                sugerencias.append("Para columnas categóricas, imputar con moda o crear categoría 'Desconocido'.")
    
    # Para análisis no supervisado, es crucial normalizar/estandarizar
    if pd.api.types.is_numeric_dtype(dtype):
        # Para columnas numéricas
        if df[columna].min() >= 0 and df[columna].max() <= 1:
            sugerencias.append("Ya parece estar normalizada (0-1).")
        else:
            # Verificar distribución para normalización
            skew = df[columna].skew()
            if abs(skew) > 1:
                sugerencias.append(f"Distribución sesgada (skew={skew:.2f}). Considere transformación logarítmica o Box-Cox.")
            
            sugerencias.append("Para análisis no supervisado, estandarizar (StandardScaler) o normalizar (MinMaxScaler).")
                
        # Verificar si podría ser categórica codificada como numérica
        if n_unique < 10 and n_unique / total < 0.05:
            sugerencias.append("Posible variable categórica. Para modelos no supervisados, considere one-hot encoding.")
            
    elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
        # Para columnas categóricas/texto
        if n_unique == 2:
            sugerencias.append("Variable binaria. Se recomienda codificación con LabelEncoder para métodos no supervisados.")
        elif n_unique <= 10:
            sugerencias.append("Se recomienda One-Hot Encoding para preservar la naturaleza categórica en análisis no supervisado.")
        else:
            sugerencias.append(f"Alta cardinalidad ({n_unique} valores únicos). Considere técnicas de reducción como agrupación de categorías poco frecuentes.")
            
    elif pd.api.types.is_datetime64_dtype(dtype):
        # Para fechas
        sugerencias.append("Extraiga características como año, mes, día, día de la semana, etc. para análisis no supervisado.")
    
    # Sugerencias específicas para análisis no supervisado
    if pd.api.types.is_numeric_dtype(dtype):
        if df[columna].std() / df[columna].mean() > 5:
            sugerencias.append("Alta variabilidad. Podría dominar el análisis no supervisado si no se estandariza correctamente.")
    
    return sugerencias

def detectar_outliers(df, column):
    """Detecta outliers en una columna numérica usando el método IQR"""
    if not pd.api.types.is_numeric_dtype(df[column]):
        return 0, 0
    
    # Calcular IQR (rango intercuartílico)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Definir límites para outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Contar outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    count = len(outliers)
    percentage = (count / len(df)) * 100
    
    return count, percentage

def encontrar_correlaciones_fuertes(corr_matrix):
    """Encuentra las correlaciones más fuertes en una matriz de correlación"""
    # Convertir a DataFrame para facilitar el manejo
    corr_df = pd.DataFrame(corr_matrix)
    
    # Crear una lista de correlaciones
    correlations = []
    for i in range(len(corr_df.columns)):
        for j in range(i+1, len(corr_df.columns)):
            col1, col2 = corr_df.columns[i], corr_df.columns[j]
            correlations.append((col1, col2, corr_df.loc[col1, col2]))
    
    # Ordenar por valor absoluto de correlación (descendente)
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Devolver las 10 correlaciones más fuertes (o menos si no hay suficientes)
    return [{"var1": corr[0], "var2": corr[1], "value": corr[2]} 
            for corr in correlations[:min(10, len(correlations))]]

def analizar_dimensionalidad(X):
    """Analiza la dimensionalidad del dataset y sugiere reducción si es necesario"""
    n_samples, n_features = X.shape
    ratio = n_samples / n_features
    
    if n_features > 20:
        if ratio < 5:
            return (f"Dataset de alta dimensionalidad ({n_features} características). "
                   f"Proporción muestras/características: {ratio:.2f}. "
                   f"Se recomienda reducción dimensional (PCA, t-SNE) para evitar la maldición de la dimensionalidad.")
        else:
            return (f"Dataset con {n_features} características. Proporción muestras/características: {ratio:.2f}. "
                   f"Se podría beneficiar de técnicas de reducción dimensional para visualización.")
    else:
        return (f"Dataset con dimensionalidad manejable ({n_features} características). "
               f"Proporción muestras/características: {ratio:.2f}. "
               f"No es estrictamente necesaria la reducción dimensional.")

@analisis_no_supervisado_bp.route('/', methods=['GET', 'POST'])
def index():
    """Página para el análisis exploratorio de datasets sin etiquetas"""
    if request.method == 'GET':
        return render_template('analisis_no_supervisado.html')
    
    # Procesar el archivo subido
    if 'file' not in request.files:
        return jsonify({'error': 'No se ha subido ningún archivo'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se ha seleccionado ningún archivo'}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'El archivo debe ser un CSV'}), 400
    
    # Obtener parámetros del formulario
    # Índice de columna a omitir (si existe)
    label_index = request.form.get('labelIndex', '')
    
    # Límite de filas
    max_rows = request.form.get('maxRows', '')
    max_rows = int(max_rows) if max_rows and max_rows.isdigit() else None
    
    try:
        # Guardar temporalmente el archivo
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
        
        # Omitir columna de etiqueta si se especificó
        columns_to_exclude = []
        if label_index and label_index.isdigit():
            label_index = int(label_index) - 1  # Convertir a índice base 0
            if label_index >= 0 and label_index < len(df.columns):
                columns_to_exclude.append(df.columns[label_index])
        
        # Extraer columnas para análisis
        X_columns = [col for col in df.columns if col not in columns_to_exclude]
        X_data = df[X_columns].copy()
        
        # Información básica del dataset
        info_buffer = io.StringIO()
        df.info(buf=info_buffer)
        info_str = info_buffer.getvalue()
        
        # Descripción estadística
        describe_html = df.describe(include='all').to_html(classes='table table-striped table-sm')
        
        # Tipos de datos por columna
        dtypes = {col: str(df[col].dtype) for col in df.columns}
        
        # Identificar columnas numéricas y categóricas
        numeric_cols = X_data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Matriz de correlación
        corr_matrix = None
        plot_corr = ""
        top_correlations = []
        
        if len(numeric_cols) > 1:
            # Calcular la matriz de correlación
            corr_matrix = X_data[numeric_cols].corr()
            
            # Visualización de la matriz de correlación
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       fmt=".2f", linewidths=0.5, ax=ax_corr)
            ax_corr.set_title('Matriz de Correlación')
            plt.tight_layout()
            plot_corr = generar_grafico_base64(fig_corr)
            
            # Encontrar correlaciones fuertes
            top_correlations = encontrar_correlaciones_fuertes(corr_matrix)
        
        # Histogramas y distribuciones
        histograms = []
        for col in numeric_cols[:min(6, len(numeric_cols))]:  # Limitar a 6 columnas
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(X_data[col].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribución de {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frecuencia')
            plt.tight_layout()
            plot_hist = generar_grafico_base64(fig)
            histograms.append({
                'title': f'Distribución de {col}',
                'image': plot_hist
            })
        
        # Visualizaciones para variables categóricas
        barplots = []
        for col in categorical_cols[:min(6, len(categorical_cols))]:  # Limitar a 6 columnas
            if X_data[col].nunique() <= 15:  # Solo si hay un número razonable de categorías
                fig, ax = plt.subplots(figsize=(10, 6))
                counts = X_data[col].value_counts()
                sns.barplot(x=counts.index, y=counts.values, ax=ax)
                ax.set_title(f'Distribución de {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Conteo')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_bar = generar_grafico_base64(fig)
                barplots.append({
                    'title': f'Distribución de {col}',
                    'image': plot_bar
                })
        
        # Scatter matrix para variables numéricas
        # Reemplazar el bloque de código de la matriz de dispersión con este código corregido:

        # Scatter matrix para variables numéricas
        scatter_matrix = None
        if len(numeric_cols) >= 2:
            try:
                # Limitar a máximo 5 variables numéricas para la matriz de dispersión
                cols_for_scatter = numeric_cols[:min(5, len(numeric_cols))]
                
                # Crear un dataframe limpio solo con las columnas que nos interesan
                scatter_df = X_data[cols_for_scatter].copy()
                # Eliminar filas con valores nulos para evitar errores
                scatter_df = scatter_df.dropna()
                
                if len(scatter_df) > 0:  # Asegurarse de que haya datos después de eliminar NaNs
                    # Usar pairplot de seaborn en lugar de scatter_matrix de pandas
                    fig = plt.figure(figsize=(12, 10))
                    g = sns.pairplot(scatter_df, diag_kind='kde', plot_kws={'alpha': 0.6})
                    plt.suptitle('Matriz de Dispersión (Variables Numéricas Principales)', y=1.02, fontsize=16)
                    plt.tight_layout()
                    scatter_matrix = generar_grafico_base64(g.fig)
                    plt.close(g.fig)  # Cerrar explícitamente la figura para liberar memoria
            except Exception as e:
                logger.error(f"Error al generar la matriz de dispersión: {str(e)}")
                scatter_matrix = None
        
        # Analizar outliers
        outliers_info = []
        for col in numeric_cols:
            outlier_count, outlier_percentage = detectar_outliers(X_data, col)
            if outlier_count > 0:
                outliers_info.append({
                    'feature': col,
                    'count': outlier_count,
                    'percentage': outlier_percentage
                })
        
        # Ordenar por porcentaje de outliers (descendente)
        outliers_info.sort(key=lambda x: x['percentage'], reverse=True)
        
        # Generar boxplots para visualizar outliers
        boxplots = []
        for col in numeric_cols[:min(6, len(numeric_cols))]:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=X_data[col].dropna(), ax=ax)
            ax.set_title(f'Boxplot de {col}')
            ax.set_xlabel(col)
            plt.tight_layout()
            plot_box = generar_grafico_base64(fig)
            boxplots.append({
                'title': f'Boxplot de {col} (para detección de outliers)',
                'image': plot_box
            })
        
        # Sugerencias de preprocesamiento
        preprocessing_suggestions = {}
        for col in df.columns:
            preprocessing_suggestions[col] = sugerir_preprocesamiento(df, col)
        
        # Análisis de dimensionalidad
        dimensionality_info = ""
        if len(numeric_cols) > 0:
            dimensionality_info = analizar_dimensionalidad(X_data[numeric_cols])
        
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
        
        # Añadir información sobre el límite de filas aplicado
        result = {
            'fileName': file.filename,
            'rowCount': len(df),
            'rowsLimited': max_rows is not None,
            'maxRowsApplied': max_rows,
            'columnCount': len(df.columns),
            'numericColumnCount': len(numeric_cols),
            'columns': columns_list,
            'dtypes': dtypes_dict,
            'info': info_str,
            'describe': describe_html,
            'tableData': table_data,
            'plotCorrelation': plot_corr,
            'histograms': histograms,
            'barplots': barplots,
            'scatterMatrix': scatter_matrix,
            'outliers': outliers_info,
            'boxplots': boxplots,
            'preprocessingSuggestions': preprocessing_suggestions,
            'missingStats': missing_stats,
            'topCorrelations': top_correlations,
            'dimensionalityInfo': dimensionality_info
        }
        
        # Usar dumps con el encoder personalizado y luego loads para serializar bien
        json_str = json.dumps(result, cls=NpEncoder)
        return json.loads(json_str)
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500