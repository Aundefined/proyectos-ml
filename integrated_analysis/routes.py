from flask import Blueprint, render_template, request, jsonify

# Crear un Blueprint para el análisis integrado
from . import integrated_analysis_bp

@integrated_analysis_bp.route('/', methods=['GET'])
def index():
    """Página para el análisis integrado que combina visualización de datasets y análisis no supervisado"""
    return render_template('integrated_analysis.html')