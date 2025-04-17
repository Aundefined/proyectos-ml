from flask import Blueprint

# Crear Blueprint
connect_four_bp = Blueprint('connect_four', __name__, 
                          url_prefix='/connect-four',
                          template_folder='templates')

# Importar rutas para que se registren con el blueprint
from . import routes