from flask import Flask, render_template, redirect, request
import os
from werkzeug.middleware.proxy_fix import ProxyFix

# Importar los blueprints
from predictor_sueldos import predictor_sueldos_bp
from visualizacion_dataset import visualizacion_dataset_bp
from analisis_no_supervisado import analisis_no_supervisado_bp
from connect_four import connect_four_bp
from integrated_analysis import integrated_analysis_bp
from clustering_clientes import clustering_clientes_bp
from predictor_seguro import predictor_seguro_bp


# Crear la aplicación Flask
app = Flask(__name__)

# Registrar los blueprints
app.register_blueprint(predictor_sueldos_bp)
app.register_blueprint(visualizacion_dataset_bp)
app.register_blueprint(analisis_no_supervisado_bp)
app.register_blueprint(connect_four_bp)
app.register_blueprint(integrated_analysis_bp)
app.register_blueprint(clustering_clientes_bp)
app.register_blueprint(predictor_seguro_bp)

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

@app.before_request
def before_request():
    if not request.is_secure and not app.debug:
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)

@app.route('/')
def home():
    """Página principal con navegación a los diferentes proyectos"""
    return render_template('home.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)