from flask import Blueprint

# Crear Blueprint
predictor_champinones_bp = Blueprint('predictor_champinones', __name__, 
                                   url_prefix='/predictor-champinones',
                                   template_folder='templates',
                                   static_folder='static',        
                                   static_url_path='/static/predictor_champinones')

# Importar rutas para que se registren con el blueprint
from . import routes