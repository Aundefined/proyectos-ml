from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf

# Crear un Blueprint para el juego de Connect Four
from . import connect_four_bp

# Cargar el modelo entrenado CNN
try:
    # Para modelos TensorFlow, usar el método de carga apropiado
    modelo = tf.keras.models.load_model('ml-models/mejor_modelo_cnn.keras')
    print("Modelo CNN cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo CNN connect four: {e}")
    # Intentar cargar como un archivo pickle (respaldo)
    try:
        with open('ml-models/modelo_cnn_connect-four.pkl', 'rb') as f:
            modelo = cloudpickle.load(f)
        print("Modelo cargado como pickle")
    except Exception as e2:
        print(f"Error al cargar el modelo mediante pickle: {e2}")
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

        # Obtener mejor jugada
        best_move = get_best_move(board_state, modelo)

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
    # La columna tiene espacio si al menos una posición en la columna está vacía
    for row in range(6):  # 6 filas
        cell_index = row * 7 + column  # Convertir coordenadas (fila, columna) a índice
        if str(cell_index) in board_state and board_state[str(cell_index)] == '.':
            return True
    
    return False


def transform_board_state_for_cnn(board_state):
    """
    Transforma el estado del tablero al formato que espera el modelo CNN
    
    Args:
        board_state: Diccionario con el estado del tablero (formato del frontend)
        
    Returns:
        Un array 1D con 42 elementos (como en el entrenamiento)
    """
    # Crear un array para las 42 celdas (6 filas × 7 columnas)
    features = np.zeros(42, dtype=np.float32)
    
    # Para cada celda del tablero (0-41)
    for i in range(42):
        # Obtenemos el valor de la celda del board_state
        if str(i) in board_state:
            cell_value = board_state[str(i)]
            # Transformar los valores como en el entrenamiento: '.' → 0, 'X' → 1, 'O' → -1
            if cell_value == 'X':
                features[i] = 1
            elif cell_value == 'O':
                features[i] = -1
            # Para '.' ya está inicializado como 0
    
    # Devolver el array como una matriz 2D (batch de 1)
    return np.array([features], dtype=np.float32)


def get_best_move(board_state, modelo):
    """
    Determina el mejor movimiento según el modelo CNN
    
    Args:
        board_state: Estado actual del tablero
        modelo: Modelo CNN entrenado para predecir movimientos
        
    Returns:
        Índice de la mejor columna (0-6)
    """
    # Convertir el estado del tablero al formato que espera el modelo CNN
    features = transform_board_state_for_cnn(board_state)
    
    # Hacemos la predicción con el modelo CNN
    try:
        # Predicción de probabilidades para cada columna
        predictions = modelo.predict(features, verbose=0)[0]
        
        # Filtrar columnas no disponibles
        available_columns = [c for c in range(7) if is_column_available(board_state, c)]
        
        if not available_columns:
            # Si no hay columnas disponibles (tablero lleno), devolvemos -1
            return -1
        
        # Seleccionar la columna disponible con mayor probabilidad
        available_predictions = [(col, predictions[col]) for col in available_columns]
        best_move = max(available_predictions, key=lambda x: x[1])[0]
        
        return best_move
    except Exception as e:
        print(f"Error en la predicción: {e}")
        # En caso de error, devolver una columna disponible aleatoria o -1
        available_columns = [c for c in range(7) if is_column_available(board_state, c)]
        return np.random.choice(available_columns) if available_columns else -1