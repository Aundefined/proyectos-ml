from flask import Blueprint

predictor_seguro_bp = Blueprint('predictor_seguro', __name__, 
    url_prefix='/predictor-seguro',
    template_folder='templates',
    static_folder='static',
    static_url_path='/static/predictor_seguro'
)

from . import routes
