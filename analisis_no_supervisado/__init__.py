from flask import Blueprint

# Crear Blueprint
analisis_no_supervisado_bp = Blueprint('analisis_no_supervisado', __name__, 
                                     url_prefix='/analisis-no-supervisado',
                                     template_folder='templates',
                                     static_folder='static',        
                                     static_url_path='/static/analisis_no_supervisado'
                                     )

# Importar rutas para que se registren con el blueprint
from . import routes