'''
Author: Zheng Wang zwang3478@gatech.edu
Date: 2023-09-20 10:10:14
LastEditors: Zheng Wang zwang3478@gatech.edu
LastEditTime: 2023-09-20 10:10:38
FilePath: /QPLoRA/utils/logger.py
'''

import logging
import os

# Define a utility method for setting the logging parameters of a logger
def get_logger(logger_name, log_dir):
    if os.path.exists(log_dir):
        raise Exception("Logging Directory {} Has Already Exists.".format(log_dir))
    else:
        os.mkdir(log_dir)

    # Get the logger with the specified name
    logger = logging.getLogger(logger_name)

    # Set the logging level of the logger to INFO
    logger.setLevel(logging.INFO)

    # Define a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # Create a console handler for outputting log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Creat a file handler for outputting log messages to the log-file
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
