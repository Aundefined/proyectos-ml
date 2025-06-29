from flask import Blueprint

recomendador_peliculas_bp = Blueprint('recomendador_peliculas', __name__, 
    url_prefix='/recomendador-peliculas',
    template_folder='templates',
    static_folder='static',
    static_url_path='/static/recomendador_peliculas'
)

from . import routes