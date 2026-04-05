from flask import Blueprint

chatbot_sechat_bp = Blueprint('chatbot_sechat', __name__,
    url_prefix='/sechat'
)

from . import routes
