from pydantic_settings import BaseSettings
import os
from typing import List, Literal, Optional

class Settings(BaseSettings):
    API_KEYS: List[str]
    ALLOWED_TOKENS: List[str]
    BASE_URL: str
    BACKEND_MODE: Optional[Literal["generic", "gemini"]] = None
    GLOBAL_THROTTLE_INTERVAL: float = 1.0  # 全局最小发送间隔（秒）
    KEY_THROTTLE_INTERVAL: float = 6.0     # 同一 Key 最小使用间隔（秒）

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        # 同时从环境变量和.env文件获取配置
        env_nested_delimiter = "__"
        extra = "ignore"

# 优先从环境变量获取,如果没有则从.env文件获取
settings = Settings(_env_file=os.getenv("ENV_FILE", ".env"))
