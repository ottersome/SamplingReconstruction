import logging
import os


def setup_logger(name: str, level=logging.INFO):
    os.makedirs("./logs", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    sh = logging.StreamHandler()
    fh = logging.FileHandler("./logs/" + name + ".log")
    sh.setLevel(level)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger
