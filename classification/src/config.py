from datetime import datetime
import os

# current date
DATE = str(datetime.now().month).zfill(2)+str(datetime.now().day).zfill(2)  

# current time
MOMENT = str(datetime.now().hour).zfill(2)+str(datetime.now().minute).zfill(2) + \
    str(datetime.now().second).zfill(2)  

# current path
SRC_PATH = os.path.dirname(os.path.abspath(__file__))