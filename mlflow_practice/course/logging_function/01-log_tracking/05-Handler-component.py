import logging

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler('app.log') # 輸出位置
file_handler.setLevel(logging.ERROR)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.debug('This is a debug message')
logger.error('This is an error message')

# 2024-06-18 12:31:57,574 - my_logger - DEBUG - This is a debug message
# 2024-06-18 12:31:57,574 - my_logger - ERROR - This is an error message