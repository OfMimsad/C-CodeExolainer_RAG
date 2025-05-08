# logger.py
import logging
import os

def setup_logger(log_file="logs/run.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

setup_logger()
