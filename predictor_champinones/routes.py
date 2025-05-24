from flask import Blueprint, render_template, request
import pandas as pd
import joblib

# Crear un Blueprint para el predictor de champiñones
from . import predictor_champinones_bp

# Cargar el modelo entrenado
try:
    model_info = joblib.load('ml-models/modelo-mushrooms.joblib')
    pipeline = model_info['pipeline']
    selected_features = model_info['selected_features']
except Exception as e:
    print(f"Error al cargar el modelo de champiñones: {e}")
    pipeline = None
    selected_features = None

# Diccionario de mapeo de valores a descripciones en español
FEATURE_MAPPINGS = {
    'odor': {
        'a': 'Almendra',
        'l': 'Anís',
        'c': 'Creosota',
        'y': 'Pescado',
        'f': 'Fétido',
        'm': 'Moho',
        'n': 'Ninguno',
        'p': 'Picante',
        's': 'Especiado'
    },
    'spore-print-color': {
        'k': 'Negro',
        'n': 'Marrón',
        'b': 'Beige',
        'h': 'Chocolate',
        'r': 'Verde',
        'o': 'Naranja',
        'u': 'Púrpura',
        'w': 'Blanco',
        'y': 'Amarillo'
    },
    'gill-size': {
        'b': 'Amplio',
        'n': 'Estrecho'
    },
    'stalk-surface-above-ring': {
        'f': 'Fibroso',
        'y': 'Escamoso',
        's': 'Liso',
        'k': 'Sedoso'
    },
    'ring-type': {
        'e': 'Evanescente',
        'f': 'Acampanado',
        'l': 'Grande',
        'n': 'Ninguno',
        'p': 'Colgante'
    },
    'stalk-surface-below-ring': {
        'f': 'Fibroso',
        'y': 'Escamoso',
        's': 'Liso',
        'k': 'Sedoso'
    },
    'population': {
        'a': 'Abundante',
        'c': 'Agrupado',
        'n': 'Numeroso',
        's': 'Disperso',
        'v': 'Varios',
        'y': 'Solitario'
    },
    'stalk-root': {
        'b': 'Bulboso',
        'c': 'Claviforme',
        'e': 'Uniforme',
        'r': 'Enraizado',
        '?': 'Desconocido'
    },
    'bruises': {
        't': 'Sí',
        'f': 'No'
    },
    'habitat': {
        'g': 'Hierba',
        'l': 'Hojas',
        'm': 'Prados',
        'p': 'Caminos',
        'u': 'Urbano',
        'w': 'Desechos',
        'd': 'Bosque'
    }
}

@predictor_champinones_bp.route('/', methods=['GET', 'POST'])
def index():
    """Página para el predictor de champiñones"""
    resultado = None
    error = None
    form_data = {feature: '' for feature in selected_features}
    
    if request.method == 'POST':
        try:
            # Validación del backend
            if not all(field in request.form and request.form[field] for field in selected_features):
                raise ValueError("Todos los campos son obligatorios")
            
            # Recoger datos del formulario
            input_data = {}
            for feature in selected_features:
                input_data[feature] = request.form.get(feature)
                form_data[feature] = input_data[feature]
            
            # Crear DataFrame completo con todas las características originales
            # (el modelo necesita todas las columnas aunque solo use las seleccionadas)
            all_features = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
                          'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 
                          'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
                          'stalk-surface-below-ring', 'stalk-color-above-ring', 
                          'stalk-color-below-ring', 'veil-type', 'veil-color', 
                          'ring-number', 'ring-type', 'spore-print-color', 
                          'population', 'habitat']
            
            # Crear diccionario con valores por defecto
            complete_data = {feature: ['n'] for feature in all_features}  # 'n' como valor por defecto
            
            # Actualizar con los valores del formulario
            for feature in selected_features:
                complete_data[feature] = [input_data[feature]]
            
            # Crear DataFrame
            df_prediccion = pd.DataFrame(complete_data)
            
            # Filtrar solo las características seleccionadas para la predicción
            df_prediccion_filtered = df_prediccion[selected_features]
            
            # Realizar predicción
            prediccion = pipeline.predict(df_prediccion_filtered)[0]
            probabilidades = pipeline.predict_proba(df_prediccion_filtered)[0]
            
            # Traducir valores a español para mostrar
            datos_traducidos = {}
            for feature, value in input_data.items():
                if feature in FEATURE_MAPPINGS and value in FEATURE_MAPPINGS[feature]:
                    datos_traducidos[feature] = FEATURE_MAPPINGS[feature][value]
                else:
                    datos_traducidos[feature] = value
            
            resultado = {
                'es_comestible': prediccion == 0,
                'clasificacion': 'Comestible' if prediccion == 0 else 'Venenoso',
                'probabilidad_comestible': round(probabilidades[0] * 100, 2),
                'probabilidad_venenoso': round(probabilidades[1] * 100, 2),
                'datos': datos_traducidos
            }
            
        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = f"Error inesperado: {str(e)}"
    
    return render_template('predictor_champinones.html', 
                         resultado=resultado, 
                         error=error, 
                         form_data=form_data,
                         feature_mappings=FEATURE_MAPPINGS)