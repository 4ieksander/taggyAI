import logging

class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter to add colors to log messages based on their severity level.
    """
    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: "\033[94m",  # Blue
        logging.INFO: "\033[92m",   # Green
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",  # Red
        logging.CRITICAL: "\033[95m",  # Magenta
    }

    def format(self, record):
        """
        Format the specified record as text.

        Args:
            record (logging.LogRecord): The log record to be formatted.

        Returns:
            str: The formatted log message with color.
        """
        log_color = self.COLORS.get(record.levelno, self.RESET)
        return f"{log_color}{super().format(record)}{self.RESET}"

# Create a console handler with a custom formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(CustomFormatter("%(asctime)s - %(levelname)s - %(message)s"))

# Create a file handler to log messages to a file
file_handler = logging.FileHandler("../logs_taggy.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Configure the root logger with the console and file handlers
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, file_handler],
)

def show_example_logs():
    """
    Generate example log messages of various severity levels.
    """
    logging.debug("This is a debug log.")
    logging.info("This is an informational log.")
    logging.warning("This is a warning log.")
    logging.error("This is an error log.")
    logging.critical("This is a critical log.")
    
def get_logger(name):
    """
    Get a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The logger instance.
    """
    return logging.getLogger(name)