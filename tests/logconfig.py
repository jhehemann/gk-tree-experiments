# logconfig.py
import logging

logger = logging.getLogger("unittest")
# Entferne alle vorhandenen Handler, damit es keine Konflikte gibt
# while logger.hasHandlers():
#     logger.removeHandler(logger.handlers[0])
for h in logger.handlers[:]:
    logger.removeHandler(h)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
