from flask import Blueprint, render_template, request
import pandas as pd

from . import predictor_seguro_bp

# Cargar el modelo entrenado de seguro
try:
    import cloudpickle
    with open('ml-models/modelo_seguro.pkl', 'rb') as f:
        modelo = cloudpickle.load(f)
except Exception as e:
    print(f"Error al cargar el modelo de seguro: {e}")
    modelo = None

@predictor_seguro_bp.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    error = None
    form_data = {
        'bmi': '',
        'age': '',
        'children': '',
        'smoker': '',
        'sex': '',
        'region': ''
    }
    if request.method == 'POST':
        # Guardar datos del formulario
        form_data = {
            'bmi': request.form.get('bmi', ''),
            'age': request.form.get('age', ''),
            'children': request.form.get('children', ''),
            'smoker': request.form.get('smoker', ''),
            'sex': request.form.get('sex', ''),
            'region': request.form.get('region', '')
        }
        try:
            # Validación de campos
            bmi = float(form_data['bmi'])
            age = int(form_data['age'])
            children = int(form_data['children'])
            smoker = form_data['smoker']
            sex = form_data['sex']
            region = form_data['region']

            if not (10 <= bmi <= 60):
                raise ValueError("El BMI debe estar entre 10 y 60.")
            if not (18 <= age <= 100):
                raise ValueError("La edad debe estar entre 18 y 100 años.")
            if not (0 <= children <= 10):
                raise ValueError("El número de hijos debe estar entre 0 y 10.")
            if smoker not in ['yes', 'no']:
                raise ValueError("Selecciona si fuma o no.")
            if sex not in ['male', 'female']:
                raise ValueError("Selecciona el sexo.")
            if region not in ['northeast', 'northwest', 'southeast', 'southwest']:
                raise ValueError("Selecciona una región válida.")

            # Crear DataFrame para el modelo
            datos = pd.DataFrame({
                'bmi': [bmi],
                'age': [age],
                'children': [children],
                'smoker': [smoker],
                'sex': [sex],
                'region': [region]
            })
            precio_predicho = modelo.predict(datos)[0]

            resultado = {
                'precio': round(precio_predicho, 2),
                'datos': {
                    'Índice de Masa Corporal (BMI)': bmi,
                    'Edad': age,
                    'Número de hijos': children,
                    '¿Fuma?': 'Sí' if smoker == 'yes' else 'No',
                    'Sexo': 'Masculino' if sex == 'male' else 'Femenino',
                    'Región': region.capitalize()
                }
            }
        except Exception as e:
            error = f"Error: {str(e)}"
    return render_template('predictor_seguro.html', resultado=resultado, error=error, form_data=form_data)
