from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    models_dir: str = "models"
    model_filename: str = "cifar10_classifier.pkl"

    class Config:
        env_file = ".env"


settings = Settings()
