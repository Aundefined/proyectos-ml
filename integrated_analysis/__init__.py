from flask import Blueprint

# Crear Blueprint
integrated_analysis_bp = Blueprint('integrated_analysis', __name__, 
                                  url_prefix='/analisis-integrado',
                                  template_folder='templates',
                                  static_folder='static',        
                                  static_url_path='/static/integrated_analysis')

# Importar rutas para que se registren con el blueprint
from . import routes