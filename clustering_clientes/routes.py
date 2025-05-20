from flask import render_template, request
import pandas as pd

# Importar el blueprint
from . import clustering_clientes_bp

@clustering_clientes_bp.route('/', methods=['GET'])
def index():
    """Página para el análisis de clustering de clientes"""
    return render_template('clustering_clientes.html')