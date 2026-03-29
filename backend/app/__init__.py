"""
MiroFish Backend - FlaskApplication工厂
"""

import os
import warnings

# 抑制 multiprocessing resource_tracker 的Warning(来自第三方库如 transformers)
# 需要在所有其他Import之前Set
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app(config_class=Config):
    """FlaskApplication工厂Function"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # SetJSONEncode: 确保中文直接显示(而不是 \uXXXX Format)
    # Flask >= 2.3 使用 app.json.ensure_ascii, 旧Version使用 JSON_AS_ASCII Config
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False
    
    # Set日志
    logger = setup_logger('mirofish')
    
    # 只在 reloader 子Process中打印StartInfo(避免 debug Pattern下打印两次)
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process
    
    if should_log_startup:
        logger.info("=" * 50)
        logger.info("MiroFish Backend Start中...")
        logger.info("=" * 50)
    
    # 启用CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register simulation process清理Function(确保Service器Close时终止所有SimulationProcess)
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("已Register simulation process清理Function")
    
    # Request日志中间件
    @app.before_request
    def log_request():
        logger = get_logger('mirofish.request')
        logger.debug(f"Request: {request.method} {request.path}")
        if request.content_type and 'json' in request.content_type:
            logger.debug(f"Request体: {request.get_json(silent=True)}")
    
    @app.after_request
    def log_response(response):
        logger = get_logger('mirofish.request')
        logger.debug(f"Response: {response.status_code}")
        return response
    
    # Register蓝图
    from .api import graph_bp, simulation_bp, report_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')
    
    # 健康检查
    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'MiroFish Backend'}
    
    if should_log_startup:
        logger.info("MiroFish Backend StartComplete")
    
    return app

