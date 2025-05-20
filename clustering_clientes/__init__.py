from flask import Blueprint

# Crear Blueprint
clustering_clientes_bp = Blueprint('clustering_clientes', __name__, 
                               url_prefix='/clustering-clientes',
                               template_folder='templates',
                               static_folder='static',        
                               static_url_path='/static/clustering_clientes')

# Importar rutas para que se registren con el blueprint
from . import routes