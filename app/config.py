"""Flask application configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

# Base directory for resolving relative paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class Config:
    """Base configuration shared by all environments."""

    SECRET_KEY = os.environ.get('SECRET_KEY')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Session security
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # ML model configuration (TFLite production champion)
    TFLITE_MODEL_PATH = os.environ.get(
        'TFLITE_MODEL_PATH',
        os.path.join(BASE_DIR, 'models', 'deployment', 'challenger_fp32.tflite'),
    )
    
    TFLITE_NUM_THREADS = int(os.environ.get('TFLITE_NUM_THREADS', '4'))

    # Haar Cascade classifier — bundled in the repository, not machine-dependent
    HAAR_CASCADE_PATH = os.path.join(
        BASE_DIR, 'app', 'static', 'haar', 'haarcascade_frontalface_default.xml',
    )

    # Support resources configuration file
    SUPPORT_RESOURCES_PATH = os.path.join(
        BASE_DIR, 'app', 'content', 'support_resources.json',
    )

    # Admin aggregation minimum group size (privacy threshold)
    MIN_AGGREGATION_GROUP_SIZE = int(os.environ.get('MIN_AGGREGATION_GROUP_SIZE', '5'))

    # Minimum confidence threshold for facial emotion scans (experimental)
    EMOTION_MIN_CONFIDENCE = float(os.environ.get('EMOTION_MIN_CONFIDENCE', '0.5'))

    # Rate limiting
    RATELIMIT_STORAGE_URI = 'memory://'

    # Maximum request body size (5MB for image payloads)
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024


class _DevelopmentConfigMeta(type):
    @property
    def SECRET_KEY(cls):
        return os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')


class DevelopmentConfig(Config, metaclass=_DevelopmentConfigMeta):
    """Development configuration."""

    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'mindease.db'),
    )

    @property
    def SECRET_KEY(self):
        return os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')


class TestingConfig(Config):
    """Testing configuration — in-memory SQLite, CSRF disabled."""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
    SECRET_KEY = 'test-secret-key'
    RATELIMIT_ENABLED = False


class _ProductionConfigMeta(type):
    @property
    def SECRET_KEY(cls):
        secret = os.environ.get('SECRET_KEY')
        if not secret:
            raise RuntimeError("SECRET_KEY environment variable is required in production")
        return secret


class ProductionConfig(Config, metaclass=_ProductionConfigMeta):
    """Production configuration."""

    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'mindease.db'),
    )
    SESSION_COOKIE_SECURE = True

    @property
    def SECRET_KEY(self):
        secret = os.environ.get('SECRET_KEY')
        if not secret:
            raise RuntimeError("SECRET_KEY environment variable is required in production")
        return secret


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}
