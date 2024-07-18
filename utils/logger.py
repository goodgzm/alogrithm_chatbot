from loguru import logger
import os
import sys

class Logger:
    def __init__(self, log_dir = "logs"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file_path = os.path.join(log_dir, "algorithm_chatbot.log")
        logger.remove()
        
        logger.add(sys.stdout, level="DEBUG")
        logger.add(log_file_path, rotation="02:00", level="DEBUG")
        self.logger = logger

LOG = Logger().logger

if __name__ == "__main__":
    log = Logger().logger

    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")

