from flask import Blueprint

# Crear Blueprint
visualizacion_dataset_bp = Blueprint('visualizacion_dataset', __name__, 
                                   url_prefix='/visualizacion-dataset',
                                   template_folder='templates')

# Importar rutas para que se registren con el blueprint
from . import routes