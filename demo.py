from us_visa.logger import logging
import sys
from us_visa.exception import USvisaException
# logging.info("welcome to our custom log")

try:
    a =2/0
except Exception as e:
     raise USvisaException(e,sys)
