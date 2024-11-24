import logging

class CustomFormatter(logging.Formatter):
    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: "\033[94m",  # Niebieski
        logging.INFO: "\033[92m",   # Zielony
        logging.WARNING: "\033[93m",  # Żółty
        logging.ERROR: "\033[91m",  # Czerwony
        logging.CRITICAL: "\033[95m",  # Magenta
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, self.RESET)
        return f"{log_color}{super().format(record)}{self.RESET}"

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(CustomFormatter("%(asctime)s - %(levelname)s - %(message)s"))

file_handler = logging.FileHandler("../taggy.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logging.basicConfig(
	level=logging.DEBUG,
	handlers=[console_handler, file_handler],
)

def show_example_logs():
	logging.debug("To jest log debug.")
	logging.info("To jest informacyjny log.")
	logging.warning("To jest log ostrzeżenia.")
	logging.error("To jest log błędu.")
	logging.critical("To jest log krytyczny.")
	
def get_logger(name):
	return logging.getLogger(name)