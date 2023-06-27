import os
import sys
import datetime


class Logger:
    def __init__(self):
        # 保存先の有無チェック
        if not os.path.isdir('./log'):
            os.makedirs('./log', exist_ok=True)
        now = datetime.datetime.now()
        filename = "./log/log" + str(now.strftime('%Y%m%d%H%M')) + ".log"
        self.console = sys.stdout
        self.file = open(filename, 'w')

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()
