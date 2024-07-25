import logging

# ■ 儲存 logging資訊
# 我們想將輸出的logging儲存的話，
# 僅須在logging.basicConfig()裡面的filename 參數設定要儲存的日誌檔名，
# 即可以將 logging 儲存起來。如未特別定義filename的參數時，預設值為 a (append 附加)，
# 表示會在原先產生的logging檔案後面添加新的logging訊息，之前的訊息不會被覆蓋，會在舊訊息之後繼續添加新訊息。
# 若想要產生的logging檔案每次都不保留舊的訊息，可以將參數改為w (write 複寫)。

def main():
    # filename: 日誌文件名稱。
    # filemode: 文件模式('w' 表示寫入模式，會覆蓋文件；'a'表示追加模式)。

    FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, filename='Log.log', filemode='a', format=FORMAT)
    logging.debug('Hello debug')
    logging.info('Hello info')
    logging.warning('Hello warning')
    logging.error('Hello error')
    logging.critical('Hello critical')

if __name__ == '__main__':
    main()



