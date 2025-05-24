from flask import Blueprint

# Crear Blueprint
clasificador_frutas_bp = Blueprint('clasificador_frutas', __name__, 
                                  url_prefix='/clasificador-frutas',
                                  template_folder='templates',
                                  static_folder='static',        
                                  static_url_path='/static/clasificador_frutas')

# Importar rutas para que se registren con el blueprint
from . import routes