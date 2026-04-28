"""
Configuration management for CS_Agent backend.
Uses Pydantic Settings for environment-based configuration.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # LLAMA (WSL) Configuration
    # 請設定為 WSL 上啟動的推理服務位址 (例如 http://127.0.0.1:11434)
    llama_api_url: str = "http://100.72.254.99:10000"
    # 可選的簡單授權 token（若你在 WSL server 實作 API key 驗證）
    llama_api_key: str | None = None
    # streaming 最大等待時間 (s)
    llama_stream_timeout: float = 600.0
    llama_request_timeout: float = 60.0
    # 預設一次生成最多 token（可視 server 支援調整）
    llama_max_tokens: int = 512

    # For backward compatibility (some modules may still reference ollama_*)
    ollama_url: str | None = None
    ollama_model: str | None = None
    ollama_debug: bool = False

    # WebSocket Configuration
    max_message_size: int = 10 * 1024  # 10KB
    history_max_length: int = 20  # Maximum number of messages to keep in history

    # Timeout Configuration (for other HTTP calls)
    request_timeout: float = 30.0
    connect_timeout: float = 5.0

    # CORS Configuration
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        # 自動從 backend/.env 載入設定，未提供時使用上方預設值
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()