from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import cloudpickle

# Crear un Blueprint para el juego de Connect Four
connect_four_bp = Blueprint('connect_four', __name__, url_prefix='/connect-four')

# Cargar el modelo entrenado
try:
    with open('modelo_connect-four.pkl', 'rb') as f:
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
    """Endpoint para hacer predicciones con el modelo"""
    try:
        # Obtener el estado del tablero desde la petición
        board_state = request.json.get('board_state', {})
        difficulty = request.json.get('difficulty', 'normal')
        
        # Transformar el estado del tablero al formato que espera el modelo
        features_dict = transform_board_state(board_state)
        
        # Convertir el diccionario a un DataFrame (el modelo fue entrenado con un DataFrame)
        features_df = pd.DataFrame([features_dict])
        
        # Asegurarnos de que solo pasamos al modelo las características que espera
        model_features = modelo.feature_names_in_
        missing_features = [feat for feat in model_features if feat not in features_df.columns]
        
        # Si faltan características, añadirlas con valor 0
        for feat in missing_features:
            features_df[feat] = 0
            
        # Asegurarnos de que las columnas están en el mismo orden que durante el entrenamiento
        features_df = features_df[model_features]
        
        # Realizar la predicción con el modelo
        if modelo is not None:
            prediction = int(modelo.predict(features_df)[0])
            
            # En modo normal, añadimos algo de aleatoriedad para hacerlo menos predecible
            # pero mantenemos la predicción del modelo como base
            if difficulty == 'normal':
                # 30% de probabilidad de elegir una columna aleatoria 
                # 70% de probabilidad de usar la predicción del modelo
                if np.random.random() < 0.3:
                    # Generar un número entre 0 y 6 (columnas posibles)
                    random_move = np.random.randint(0, 7)
                    # Comprobar si la columna está disponible
                    if is_column_available(board_state, random_move):
                        prediction = random_move
            
            # Validar que la predicción sea una columna válida y disponible (0-6)
            if 0 <= prediction <= 6 and is_column_available(board_state, prediction):
                return jsonify({'success': True, 'move': prediction})
            else:
                # Si el modelo da una predicción inválida, elegir una columna disponible al azar
                available_columns = [col for col in range(7) if is_column_available(board_state, col)]
                if available_columns:
                    return jsonify({'success': True, 'move': np.random.choice(available_columns)})
                else:
                    return jsonify({'success': False, 'error': 'No hay columnas disponibles'})
        else:
            return jsonify({'success': False, 'error': 'Modelo no disponible'}), 500
    except Exception as e:
        print(f"Error en la predicción: {str(e)}")
        # En caso de error, intentar devolver una columna válida
        try:
            available_columns = [col for col in range(7) if is_column_available(board_state, col)]
            if available_columns:
                return jsonify({'success': True, 'move': np.random.choice(available_columns)})
            else:
                return jsonify({'success': False, 'error': 'No hay columnas disponibles'})
        except:
            # Si todo falla, devolver columna del medio como último recurso
            return jsonify({'success': True, 'move': 3})

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