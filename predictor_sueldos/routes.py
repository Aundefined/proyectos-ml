from flask import Blueprint, render_template, request
import pandas as pd
import joblib
import math

# Crear un Blueprint para el predictor de sueldos
from . import predictor_sueldos_bp

# Cargar el modelo entrenado
try:
    import cloudpickle
    with open('ml-models/modelo_sueldos.pkl', 'rb') as f:
        modelo = cloudpickle.load(f)
except Exception as e:
    print(f"Error al cargar el modelo sueldos: {e}")
    modelo = None

@predictor_sueldos_bp.route('/', methods=['GET', 'POST'])
def index():
    """Página para el predictor de sueldos"""
    resultado = None
    error = None
    form_data = {
        'edad': '',
        'experiencia': '',
        'horas': '',
        'educacion': '',
        'industria': '',
        'ubicacion': '',
        'genero': ''
    }
    
    if request.method == 'POST':
        # Recoger los datos del formulario
        form_data = {
            'edad': request.form.get('edad', ''),
            'experiencia': request.form.get('experiencia', ''),
            'horas': request.form.get('horas', ''),
            'educacion': request.form.get('educacion', ''),
            'industria': request.form.get('industria', ''),
            'ubicacion': request.form.get('ubicacion', ''),
            'genero': request.form.get('genero', '')
        }
        
        try:
            # Validación del backend
            if not all(field in request.form and request.form[field] for field in 
                       ['edad', 'experiencia', 'horas', 'educacion', 'industria', 'ubicacion', 'genero']):
                raise ValueError("Todos los campos son obligatorios")
            
            # Recoger datos del formulario
            edad = int(request.form.get('edad'))
            experiencia = int(request.form.get('experiencia'))
            horas = int(request.form.get('horas'))
            educacion = request.form.get('educacion')
            industria = request.form.get('industria')
            ubicacion = request.form.get('ubicacion')
            genero = request.form.get('genero')
            
            # Validaciones adicionales
            if not (22 <= edad <= 65):
                raise ValueError("La edad debe estar entre 22 y 65 años")
            if not (0 <= experiencia <= 47):
                raise ValueError("La experiencia debe estar entre 0 y 47 años")
            if not (30 <= horas <= 60):
                raise ValueError("Las horas semanales deben estar entre 30 y 60")
            if experiencia > (edad - 22):
                raise ValueError("La experiencia no puede ser mayor que la edad menos 22 años")
            if educacion not in ['Licenciatura', 'Máster', 'Doctorado']:
                raise ValueError("Nivel educativo no válido")
            if industria not in ['Tecnología', 'Finanzas', 'Salud', 'Manufactura', 'Retail']:
                raise ValueError("Industria no válida")
            if ubicacion not in ['EE.UU.', 'Europa', 'Latinoamérica']:
                raise ValueError("Ubicación no válida")
            if genero not in ['Masculino', 'Femenino', 'Otro']:
                raise ValueError("Género no válido")
            
            # Crear DataFrame con el mismo formato que se usó para entrenar
            nuevo_dato = {
                'Edad': [edad],
                'Experiencia_Anios': [experiencia],
                'Horas_Semanales': [horas],
                'Nivel_Educativo_Doctorado': [1 if educacion == 'Doctorado' else 0],
                'Nivel_Educativo_Licenciatura': [1 if educacion == 'Licenciatura' else 0],
                'Nivel_Educativo_Máster': [1 if educacion == 'Máster' else 0],
                'Industria_Finanzas': [1 if industria == 'Finanzas' else 0],
                'Industria_Manufactura': [1 if industria == 'Manufactura' else 0],
                'Industria_Retail': [1 if industria == 'Retail' else 0],
                'Industria_Salud': [1 if industria == 'Salud' else 0],
                'Industria_Tecnología': [1 if industria == 'Tecnología' else 0],
                'Ubicación_EE.UU.': [1 if ubicacion == 'EE.UU.' else 0],
                'Ubicación_Europa': [1 if ubicacion == 'Europa' else 0],
                'Ubicación_Latinoamérica': [1 if ubicacion == 'Latinoamérica' else 0],
                'Género_Masculino': [1 if genero == 'Masculino' else 0],
                'Género_Otro': [1 if genero == 'Otro' else 0]
            }
            
            nuevo_dato_df = pd.DataFrame(nuevo_dato)
            
            # Realizar predicción
            prediccion_log = modelo.predict(nuevo_dato_df)
            # Invertir la transformación logarítmica para obtener el sueldo en escala original
            sueldo_predicho = math.exp(prediccion_log[0]) - 1
            
            resultado = {
                'sueldo': round(sueldo_predicho, 2),
                'datos': {
                    'Edad': edad,
                    'Experiencia': experiencia,
                    'Horas semanales': horas,
                    'Educación': educacion,
                    'Industria': industria,
                    'Ubicación': ubicacion,
                    'Género': genero
                }
            }
        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = f"Error inesperado: {str(e)}"
        
    return render_template('predictor_sueldos.html', 
                           resultado=resultado, 
                           error=error, 
                           form_data=form_data)