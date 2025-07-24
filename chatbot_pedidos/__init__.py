from flask import Blueprint

chatbot_pedidos_bp = Blueprint('chatbot_pedidos', __name__, 
    url_prefix='/chatbot-pedidos',
    template_folder='templates',
    static_folder='static',
    static_url_path='/static/chatbot_pedidos'
)

from . import routes