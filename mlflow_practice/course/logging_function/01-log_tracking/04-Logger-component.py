import logging

# level: 設置最低的日誌級別。低於這個級別的日誌消息會被忽略。常見級別有 DEBUG, INFO, WARNING, ERROR, CRITICAL。
# format: 設置日誌輸出格式。
# %(asctime)s: 記錄時間。
# %(name)s: Logger 的名稱。
# %(levelname)s: 日誌級別。
# %(message)s: 日誌消息。

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')
