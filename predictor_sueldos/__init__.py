from flask import Blueprint

# Crear Blueprint
predictor_sueldos_bp = Blueprint('predictor_sueldos', __name__, 
                               url_prefix='/predictor-sueldos',
                               template_folder='templates',
                               static_folder='static',        
                               static_url_path='/static/predictor_sueldos')

# Importar rutas para que se registren con el blueprint
from . import routes