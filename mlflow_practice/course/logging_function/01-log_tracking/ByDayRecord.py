#！coding:utf-8
import logging
import logging.handlers
import datetime, time

#logging    初始化工作
logger = logging.getLogger("zjlogger")
logger.setLevel(logging.DEBUG)

# 添加TimedRotatingFileHandler
# 定义一个1秒换一次log文件的handler
# 保留3个旧log文件
rf_handler = logging.handlers.TimedRotatingFileHandler(filename="all.log",when='S',interval=1,\
                                                       backupCount=3)
rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

#在控制台打印日志
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger.addHandler(rf_handler)
logger.addHandler(handler)

while True:
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warning message')
    logger.error('error message')
    logger.critical('critical message')
    time.sleep(1)