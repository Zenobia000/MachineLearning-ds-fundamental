import logging

# 在logging中，也可以透過3種方法來記錄Exception的錯誤訊息。

# 1. 直接輸出Exception訊息。
def main_sol1():
    logging.basicConfig(level=logging.DEBUG)
    try:
        llllllllogging.debug('Hello Debug')  # 定義錯誤
        logging.debug('Hello Debug')
        logging.info('Hello info')
        logging.warning('Hello WARNING')
        logging.error('Hello ERROR')
        logging.critical('Hello CRITICAL')
    except Exception as e:
        logging.error('Exception ERROR => ' + str(e))



# 2. 在 logging.error() 加上 exc_info 參數，並將該參數設為 True，就可以紀錄 Exception。
def main_sol2():
    logging.basicConfig(level=logging.DEBUG)
    try:
        llllllllogging.debug('Hello Debug') # 定義錯誤
    except Exception as e:
        logging.error("Catch an exception.", exc_info=True)



# 3. 若要在 logging 內紀錄 exception 訊息，可使用 logging.exception()，
# 它會將 exception 添加至訊息中，此方法的等級為 ERROR，也就是說 logging.exception() 就等同於 logging.error(exc_info=True)

def main_sol3():
    logging.basicConfig(level=logging.DEBUG)
    try:
        llllllllogging.debug('Hello Debug') # 定義錯誤
    except Exception as e:
        logging.exception('Catch an exception.')



if __name__ == '__main__':
    main_sol1()
    # main_sol2()
    # main_sol3()



