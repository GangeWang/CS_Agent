"""
Configuration management for CS_Agent backend.
Uses Pydantic Settings for environment-based configuration.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Ollama Configuration
    ollama_model: str = "cs-agent-v17"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_debug: bool = False
    
    # WebSocket Configuration
    max_message_size: int = 10 * 1024  # 10KB
    history_max_length: int = 20  # Maximum number of messages to keep in history
    
    # Timeout Configuration
    request_timeout: float = 30.0
    connect_timeout: float = 5.0
    
    # CORS Configuration
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    class Config:
        # 自動從 backend/.env 載入設定，未提供時使用上方預設值
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
# 於 import 時建立單例設定，其他模組可直接 from app.config import settings 使用
settings = Settings()
