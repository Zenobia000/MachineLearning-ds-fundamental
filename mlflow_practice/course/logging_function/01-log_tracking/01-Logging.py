import logging

# Logging把等級分為六個等級，
# 包含：NOTSET、DEBUG、INFO、WARNING、ERROR、CRITICAL。
# 如下列表每個等級都有一個對應的數值，而Loggin只會紀錄數值大於設定值以上的訊息。
# 例如：設定等級為WARNING，日後記錄數值就不會紀錄對應數值低於30以下的訊息。

print("查詢 log 等級對應號碼")
# 查詢 log 等級對應號碼
print(logging.NOTSET)   # 輸出為：0
print(logging.DEBUG)    # 輸出為：10
print(logging.INFO)     # 輸出為：20
print(logging.WARNING)  # 輸出為：30
print(logging.ERROR)    # 輸出為：40
print(logging.CRITICAL) # 輸出為：50


print("查詢 log 號碼對應等級")
# 查詢 log 號碼對應等級
print(logging.getLevelName(0))    # 輸出為：NOTSET
print(logging.getLevelName(10))   # 輸出為：DEBUG
print(logging.getLevelName(20))   # 輸出為：INFO
print(logging.getLevelName(30))   # 輸出為：WARNING
print(logging.getLevelName(40))   # 輸出為：ERROR
print(logging.getLevelName(50))   # 輸出為：CRITICAL


# Logging輸出
# 介紹上述的Logging分級後，為大家說明一下Logging分級後的輸出，假設我們預設當前的Logging輸出為WARNING等級，分別輸出DEBUG、INFO、WARNING、ERROR、CRITICAL ，並查看其結果。
def main():
## ===== 定義Logging等級為WARNING  ===== ##

    # 預設的訊息輸出格式只有 levelname、name、message。
    logging.basicConfig(level=logging.WARNING)
    logging.debug('Hello Debug')
    logging.info('Hello info')
    logging.warning('Hello WARNING')
    logging.error('Hello ERROR')
    logging.critical('Hello CRITICAL')

def main_format():
    # 定義輸出格式
    FORMAT = '%(asctime)s %(filename)s %(levelname)s:%(message)s'
    # Logging初始設定 + 上定義輸出格式
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    logging.info('Hello python')


if __name__ == '__main__':
    # main()
    main_format()
