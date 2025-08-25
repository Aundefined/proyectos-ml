from flask import Blueprint, render_template, request, flash
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import tempfile

# Crear un Blueprint para el clasificador de frutas
from . import clasificador_frutas_bp

# Configuración
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Cargar el modelo entrenado
try:
    modelo = tf.keras.models.load_model('ml-models/mejor_modelo_frutas_transfer.h5')
    print("Modelo de frutas cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo de frutas: {e}")
    modelo = None

# Diccionario de mapeo de clases
CLASE_MAPPING = {
    'fresh_peaches_done': 'Melocotones Frescos',
    'fresh_pomegranates_done': 'Granadas Frescas',
    'fresh_strawberries_done': 'Fresas Frescas',
    'rotten_peaches_done': 'Melocotones Podridos',
    'rotten_pomegranates_done': 'Granadas Podridas',
    'rotten_strawberries_done': 'Fresas Podridas'
}

# Orden de las clases según el modelo
CLASE_INDICES = {
    'fresh_peaches_done': 0,
    'fresh_pomegranates_done': 1,
    'fresh_strawberries_done': 2,
    'rotten_peaches_done': 3,
    'rotten_pomegranates_done': 4,
    'rotten_strawberries_done': 5
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    """Función para predecir la clase de una imagen"""
    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Predecir
    prediction = modelo.predict(img_array)
    
    # Obtener la clase predicha
    predicted_idx = np.argmax(prediction)
    
    # Mapear índice a nombre de clase
    idx_to_class = {v: k for k, v in CLASE_INDICES.items()}
    predicted_class = idx_to_class[predicted_idx]
    
    # Obtener todas las probabilidades
    probabilidades = {}
    for clase, idx in CLASE_INDICES.items():
        probabilidades[CLASE_MAPPING[clase]] = float(prediction[0][idx] * 100)
    
    return {
        'clase_predicha': CLASE_MAPPING[predicted_class],
        'clase_original': predicted_class,
        'confianza': float(np.max(prediction) * 100),
        'probabilidades': probabilidades,
        'es_fresca': 'fresh' in predicted_class
    }

@clasificador_frutas_bp.route('/', methods=['GET', 'POST'])
def index():
    """Página principal del clasificador de frutas"""
    resultado = None
    error = None
    
    if request.method == 'POST' and modelo is not None:
        try:
            # Verificar si se subió un archivo
            if 'imagen' not in request.files:
                raise ValueError("No se seleccionó ningún archivo")
            
            file = request.files['imagen']
            
            # Verificar si se seleccionó un archivo
            if file.filename == '':
                raise ValueError("No se seleccionó ningún archivo")
            
            # Verificar el tipo de archivo
            if not allowed_file(file.filename):
                raise ValueError("Tipo de archivo no permitido. Use PNG, JPG, JPEG, GIF o BMP")
            
            # Verificar el tamaño del archivo
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > MAX_FILE_SIZE:
                raise ValueError(f"El archivo es demasiado grande. Máximo {MAX_FILE_SIZE/1024/1024}MB")
            
            # Guardar temporalmente el archivo
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                file.save(tmp_file.name)
                tmp_filename = tmp_file.name
            
            try:
                # Realizar la predicción
                resultado = predict_image(tmp_filename)
                resultado['nombre_archivo'] = secure_filename(file.filename)
            finally:
                # Eliminar el archivo temporal
                if os.path.exists(tmp_filename):
                    os.unlink(tmp_filename)
                    
        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = f"Error al procesar la imagen: {str(e)}"
            print(f"Error detallado: {e}")
    
    elif modelo is None:
        error = "El modelo no está disponible. Por favor, contacte al administrador."
    
    return render_template('clasificador_frutas.html', 
                         resultado=resultado, 
                         error=error,
                         clases=CLASE_MAPPING)