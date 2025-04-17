from flask import Blueprint

# Crear Blueprint
visualizacion_dataset_bp = Blueprint('visualizacion_dataset', __name__, 
                                   url_prefix='/visualizacion-dataset',
                                   template_folder='templates',
                                   static_folder='static',        
                                   static_url_path='/static/visualizacion_dataset')

# Importar rutas para que se registren con el blueprint
from . import routes