from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import cloudpickle
import joblib

# Crear un Blueprint para el juego de Connect Four
from . import connect_four_bp

# Cargar el modelo entrenado
try:
    import cloudpickle
    with open('ml-models/modelo_connect-four.pkl', 'rb') as f:
        modelo = cloudpickle.load(f)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    modelo = None


@connect_four_bp.route('/', methods=['GET'])
def index():
    """Página para el juego de Connect Four"""
    return render_template('connect_four.html')

@connect_four_bp.route('/predict', methods=['POST'])
def predict():
    try:
        board_state = request.json.get('board_state', {})
        difficulty = request.json.get('difficulty', 'normal')

        if modelo is None:
            return jsonify({'success': False, 'error': 'Modelo no disponible'}), 500

        model_features = modelo.feature_names_in_

        # Obtener mejor jugada
        best_move = get_best_move(board_state, modelo, model_features)

        # Aleatoriedad en modo normal
        if difficulty == 'normal' and np.random.random() < 0.3:
            available_columns = [c for c in range(7) if is_column_available(board_state, c)]
            if available_columns:
                best_move = np.random.choice(available_columns)

        return jsonify({'success': True, 'move': int(best_move)})
    except Exception as e:
        print(f"Error en la predicción: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


def is_column_available(board_state, column):
    """
    Verifica si una columna tiene espacio disponible
    """
    # Chequear si la posición superior de la columna está vacía
    for row in range(6):  # 6 filas
        cell_index = row * 7 + column  # Convertir coordenadas (fila, columna) a índice
        if str(cell_index) in board_state and board_state[str(cell_index)] == '.':
            return True
    
    return False


def transform_board_state(board_state):
    """
    Transforma el estado del tablero al formato que espera el modelo
    
    Args:
        board_state: Diccionario con el estado del tablero (formato del frontend)
        
    Returns:
        Un DataFrame con las características en el formato que espera el modelo
    """
    # Creamos un diccionario para almacenar las características one-hot encoding
    features = {}
    
    # Para cada celda del tablero (0-41)
    for i in range(42):
        # Inicializamos todos los valores posibles como False
        features[f'cell_{i}_.'] = False
        features[f'cell_{i}_O'] = False
        features[f'cell_{i}_X'] = False
        
        # Obtenemos el valor de la celda si existe en board_state
        if str(i) in board_state:
            cell_value = board_state[str(i)]
            # Marcamos como True la característica correspondiente
            features[f'cell_{i}_{cell_value}'] = True
        else:
            # Si no hay valor, asumimos que está vacía ('.')
            features[f'cell_{i}_.'] = True
    
    # Convertimos a DataFrame para asegurar el formato correcto
    return pd.DataFrame([features])


def get_best_move(board_state, modelo, model_features=None):
    """
    Determina el mejor movimiento según el modelo
    
    Args:
        board_state: Estado actual del tablero
        modelo: Modelo entrenado para predecir movimientos
        model_features: Lista opcional de características que espera el modelo
        
    Returns:
        Índice del mejor movimiento (0-41)
    """
    # Obtenemos las características en el formato esperado (ya como DataFrame)
    features_df = transform_board_state(board_state)
    
    debug_dataframe(features_df)
    
    # Aseguramos que todas las columnas estén presentes y en el orden correcto
    if model_features is not None:
        # Añadimos cualquier columna faltante
        for feature in model_features:
            if feature not in features_df.columns:
                features_df[feature] = False
        # Ordenamos las columnas para que coincidan con el modelo
        features_df = features_df[model_features]
    
    # Hacemos la predicción
    try:
        move = modelo.predict(features_df)[0]
        moveInt= int(move)
        return moveInt
    except Exception as e:
        print(f"Error en la predicción: {e}")
        print(f"Forma de los datos: {features_df.shape}")
        print(f"Primeras columnas: {list(features_df.columns)[:5]}")
        # En caso de error, devolver un movimiento por defecto o gestionar el error
        return -1  # Código de error















    
# Opciones de depuración del df.    
# Opción 1: Convierte a HTML y guárdalo en un archivo
def debug_dataframe(df, filename="debug_df.html"):
    with open(filename, "w") as f:
        f.write(df.to_html())
    print(f"DataFrame guardado en {filename}")

# Opción 2: Imprime un resumen más completo
def print_df_debug(df):
    print(f"Shape: {df.shape}")
    print("\nPrimeras 5 filas:")
    print(df.head().to_string())
    print("\nColumnas:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
    print("\nValores de la primera fila:")
    for col in df.columns:
        print(f"{col}: {df.iloc[0][col]}")

# Opción 3: Guarda en CSV para revisar en Excel
def save_debug_csv(df, filename="debug_df.csv"):
    df.to_csv(filename)
    print(f"DataFrame guardado en {filename}")