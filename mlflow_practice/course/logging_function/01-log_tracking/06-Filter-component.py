import logging

class MyFilter(logging.Filter):
    def filter(self, record):
        return 'special' in record.msg

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.addFilter(MyFilter())

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.debug('This is a special debug message')
logger.debug('This is a regular debug message')

