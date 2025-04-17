from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import cloudpickle

# Crear un Blueprint para el juego de Connect Four
from . import connect_four_bp

# Cargar el modelo entrenado
try:
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
        Un diccionario con las características en el formato que espera el modelo
    """
    # Primero, convertimos el estado del tablero a una representación en matriz 6x7
    board_matrix = [['.'] * 7 for _ in range(6)]
    
    # Llenamos la matriz con el estado actual
    for i in range(42):
        row = i // 7
        col = i % 7
        if str(i) in board_state:
            board_matrix[row][col] = board_state[str(i)]
    
    # Ahora convertimos esta matriz al formato que espera el modelo
    # Siguiendo exactamente el formato del entrenamiento en el notebook:
    # Debemos crear columnas como 'cell_0', 'cell_1', etc. 
    # donde cada celda tiene un valor '.', 'O', o 'X'
    raw_features = {}
    for i in range(42):
        row = i // 7
        col = i % 7
        raw_features[f'cell_{i}'] = board_matrix[row][col]
    
    # Creamos un DataFrame con estas características
    df = pd.DataFrame([raw_features])
    
    # Aplicamos pd.get_dummies para obtener exactamente el mismo formato que en el entrenamiento
    features_df = pd.get_dummies(df, drop_first=False)
    
    # Convertimos el DataFrame a diccionario
    return features_df.iloc[0].to_dict()


def get_best_move(board_state, modelo, model_features):
    import pandas as pd
    import numpy as np

    def transform_state(state):
        board_matrix = [['.'] * 7 for _ in range(6)]
        for i in range(42):
            row = i // 7
            col = i % 7
            board_matrix[row][col] = state.get(str(i), '.')
        raw_features = {f'cell_{i}': board_matrix[i // 7][i % 7] for i in range(42)}
        df = pd.DataFrame([raw_features])
        df_encoded = pd.get_dummies(df, drop_first=False)
        for feat in model_features:
            if feat not in df_encoded.columns:
                df_encoded[feat] = 0
        return df_encoded[model_features]

    def get_next_state(state, col, token='X'):
        new_state = state.copy()
        for row in reversed(range(6)):
            idx = row * 7 + col
            if new_state[str(idx)] == '.':
                new_state[str(idx)] = token
                break
        return new_state

    best_move = None
    best_score = -float('inf')

    for col in range(7):
        if not is_column_available(board_state, col):
            continue
        next_state = get_next_state(board_state, col)
        features = transform_state(next_state)
        try:
            score = modelo.predict_proba(features).max()
        except:
            score = 0
        if score > best_score:
            best_score = score
            best_move = col

    return best_move if best_move is not None else np.random.choice([c for c in range(7) if is_column_available(board_state, c)])
