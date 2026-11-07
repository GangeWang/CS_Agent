import os
import sys

# 讓 pytest 從 backend/ 目錄執行時可直接 import app 套件
sys.path.insert(0, os.path.dirname(__file__))
