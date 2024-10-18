import logging

# 创建并配置 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台 handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 将 handler 添加到 logger
logger.addHandler(console_handler)

# 示例日志输出
logger.info("Logging system is working correctly")
