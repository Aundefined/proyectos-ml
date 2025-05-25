from flask import Blueprint

sobre_mi_bp = Blueprint('sobre_mi', __name__, 
                       url_prefix='/sobre-mi',
                       template_folder='templates',
                       static_folder='static',        
                       static_url_path='/static/sobre_mi')

from . import routes