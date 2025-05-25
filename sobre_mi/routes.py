from . import routes


from flask import render_template

# Importar el blueprint desde __init__.py
from . import sobre_mi_bp

@sobre_mi_bp.route('/')
def index():
    """Página sobre mí con currículum interactivo"""
    return render_template('sobre_mi.html')