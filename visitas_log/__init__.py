from flask import Blueprint

visitas_log_bp = Blueprint('visitas_log', __name__, 
                          url_prefix='',
                          template_folder='templates',
                          static_folder='static',
                          static_url_path='/static/visitas_log')

from . import routes