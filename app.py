from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Cargamos el modelo entrenado
modelo = joblib.load('modelo_sueldos.pkl')

@app.route('/')
def home():
    """Página principal con navegación a los diferentes proyectos"""
    return render_template('home.html')

@app.route('/predictor-sueldos', methods=['GET', 'POST'])
def predictor_sueldos():
    """Página para el predictor de sueldos"""
    resultado = None
    error = None
    
    if request.method == 'POST':
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
                raise ValueError("La experiencia no puede ser mayor que la edad menos 18 años")
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
            sueldo_predicho = np.expm1(prediccion_log[0])
            
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
        
    return render_template('predictor_sueldos.html', resultado=resultado, error=error)

if __name__ == '__main__':
    app.run(debug=True)