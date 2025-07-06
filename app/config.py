from pydantic_settings import BaseSettings
import os
from typing import List

class Settings(BaseSettings):
    API_KEYS: List[str]
    ALLOWED_TOKENS: List[str]
    BASE_URL: str
    THROTTLE_INTERVAL: int = 3  # 新增：默认值为3秒

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        # 同时从环境变量和.env文件获取配置
        env_nested_delimiter = "__"
        extra = "ignore"

# 优先从环境变量获取,如果没有则从.env文件获取
settings = Settings(_env_file=os.getenv("ENV_FILE", ".env"))
