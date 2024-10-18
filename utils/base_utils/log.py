#
# Log file utilities. Call the configure_log_handler() with appropriate log file
# name before logging anything.
#

from os import path, remove

import logging
import traceback

# Create the logger
logger = logging.getLogger(__name__)
# Set the logging level to DEBUG, such that all level messages are logged.
logger.setLevel(logging.DEBUG)

def configure_log_handler(log_file):
  """
  Configures the log handler. Mandatory to be called before logging.

  Args:
    log_file (str): Path/to/log/file/name.log
  """
  if not log_file:
    raise Exception("No log_file found! Assign a log file path to it.")

  # If applicable, delete the existing log file to generate a fresh log file
  # during each execution
  if path.isfile(log_file):
    remove(log_file)

  # Create handler for logging the messages to a log file.
  log_handler = logging.FileHandler(log_file)
  log_handler.setLevel(logging.DEBUG)

  # Set the format of the log.
  log_formatter = logging.Formatter(
      "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

  # Add the Formatter to the Handler
  log_handler.setFormatter(log_formatter)

  # Add stream handler.
  logger.addHandler(logging.StreamHandler())

  # Add the Handler to the Logger
  logger.addHandler(log_handler)
  logger.info('Completed configuring logger()!')

  return logger

def DEBUG(msg):
  if len(logger.handlers) == 0:
    print(msg)
  else:
    logger.debug(msg)

def INFO(msg):
  if len(logger.handlers) == 0:
    print(msg)
  else:
    logger.info(msg)

def WARN(msg):
  if len(logger.handlers) == 0:
    print(msg)
  else:
    logger.warning(msg)
  excp = traceback.format_exc()
  if excp != "NoneType: None":
    logger.warning(excp)

def ERROR(msg):
  if len(logger.handlers) == 0:
    print(msg)
  else:
    logger.error(msg)
  excp = traceback.format_exc()
  if excp != "NoneType: None":
    logger.error(excp)

def RESET():
  handlers = logger.handlers.copy()
  for handler in handlers:
    handler.acquire()
    handler.flush()
    handler.close()
    handler.release()
    logger.removeHandler(handler)
