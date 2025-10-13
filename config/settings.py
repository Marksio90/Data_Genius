from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_model: str = "gpt-4o-mini" 

    model_config = SettingsConfigDict(
        env_file=".env",  # tylko dla innych ustawień, nie kluczy API
        env_prefix="",    # bez prefixu (możesz dodać np. "APP_")
        extra="ignore",
    )

settings = Settings()