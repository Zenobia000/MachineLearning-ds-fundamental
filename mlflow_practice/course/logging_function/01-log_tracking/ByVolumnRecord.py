#!coding:utf-8
#!/usr/bin/env python
import time
import logging
import logging.handlers
# logging初始化工作
logging.basicConfig()
# myapp的初始化工作
myapp = logging.getLogger('myapp')
myapp.setLevel(logging.INFO)
# 写入文件，如果文件超过100个Bytes，仅保留5个文件。
handler = logging.handlers.RotatingFileHandler(
              'logs/myapp.log', maxBytes=100, backupCount=5)

myapp.addHandler(handler)

while True:
    time.sleep(0.01)
    myapp.info("file test")