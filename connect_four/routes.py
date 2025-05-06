from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf

# Crear un Blueprint para el juego de Connect Four
from . import connect_four_bp

# Esta clase es un adaptador para hacer que TFLite funcione como tu modelo RL
class TFLiteModelAdapter:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def predict(self, input_data, verbose=0):
        # Asegurar que los datos tienen el formato correcto
        input_data = np.array(input_data, dtype=np.float32)
        
        # Establecer los datos de entrada
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Ejecutar la inferencia
        self.interpreter.invoke()
        
        # Obtener resultados
        return self.interpreter.get_tensor(self.output_details[0]['index'])

# Cargar el modelo entrenado con RL
try:
    # Cargar el modelo TFLite
    modelo = TFLiteModelAdapter('ml-models/connect4_ep8500_win44.tflite')
    print("Modelo TFLite RL cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo TFLite RL: {e}")
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
        best_move = get_best_move(board_state, modelo, difficulty)

        return jsonify({'success': True, 'move': int(best_move)})
    except Exception as e:
        print(f"Error en la predicción: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


def is_column_available(board_state, column):
    """
    Verifica si una columna tiene espacio disponible
    """
    for row in range(6):  # 6 filas
        cell_index = row * 7 + column  # Convertir coordenadas (fila, columna) a índice
        if str(cell_index) in board_state and board_state[str(cell_index)] == '.':
            return True
    
    return False


def transform_board_state_for_rl(board_state):
    """
    Transforma el estado del tablero al formato que espera el modelo RL (matriz 6x7)
    
    Args:
        board_state: Diccionario con el estado del tablero (formato del frontend)
        
    Returns:
        Una matriz numpy 6x7 con el estado del tablero
    """
    # Crear matriz 6x7 para el tablero (0=vacío, 1=jugador, -1=IA)
    board = np.zeros((6, 7), dtype=np.float32)
    
    # Para cada celda del tablero
    for i in range(42):
        row = i // 7
        col = i % 7
        
        if str(i) in board_state:
            cell_value = board_state[str(i)]
            # Transformar los valores: '.' → 0, 'X' → 1, 'O' → -1
            # Nota: En el entrenamiento RL, PLAYER_1 (agente) = 1, PLAYER_2 (oponente) = -1
            # Pero en la app web, el jugador es 'O' y la IA es 'X'
            # Así que invertimos los valores para mantener consistencia con el entrenamiento
            if cell_value == 'X':  # IA (modelo)
                board[row][col] = 1
            elif cell_value == 'O':  # Jugador humano
                board[row][col] = -1
            # Para '.' ya está inicializado como 0
    
    return board


def get_best_move(board_state, modelo, difficulty):
    """
    Determina el mejor movimiento según el modelo RL
    
    Args:
        board_state: Estado actual del tablero
        modelo: Modelo RL entrenado
        difficulty: Nivel de dificultad seleccionado
        
    Returns:
        Índice de la mejor columna (0-6)
    """
    # Obtener columnas disponibles
    available_columns = [c for c in range(7) if is_column_available(board_state, c)]
    
    if not available_columns:
        # Si no hay columnas disponibles, devolver -1
        return -1
    
    # En modo normal, añadimos aleatoriedad
    if difficulty == 'normal' and np.random.random() < 0.3:
        return np.random.choice(available_columns)
    
    try:
        # Convertir el estado del tablero al formato que espera el modelo RL
        board = transform_board_state_for_rl(board_state)
        
        # Hacer predicción con el modelo
        q_values = modelo.predict(np.expand_dims(board, axis=0), verbose=0)[0]
        
        # Aplicar máscara para movimientos inválidos (asignar valores muy bajos)
        for col in range(7):
            if col not in available_columns:
                q_values[col] = -float('inf')
        
        # Seleccionar columna con mayor valor Q
        best_move = np.argmax(q_values)
        
        return best_move
    except Exception as e:
        print(f"Error en la predicción: {e}")
        # En caso de error, elegir una columna aleatoria
        return np.random.choice(available_columns)