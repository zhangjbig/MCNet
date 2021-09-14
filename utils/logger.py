import os
import logging
def init_log_config(log_path, log_name):
    """
    初始化日志相关配置
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, log_name)
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
