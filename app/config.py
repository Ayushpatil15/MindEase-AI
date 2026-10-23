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


def _normalize_database_url(url, default):
    """Normalize database connection URL for SQLAlchemy with explicit psycopg2 driver.

    Render and legacy PaaS providers inject PostgreSQL URLs starting with 'postgres://'
    or 'postgresql://'. Without an explicit driver specification ('postgresql+psycopg2://'),
    SQLAlchemy in certain environments attempts to load the Psycopg 3 driver ('psycopg')
    instead of the installed 'psycopg2-binary' ('psycopg2'), triggering:
    ModuleNotFoundError: No module named 'psycopg'.

    This function normalizes 'postgres://', 'postgresql://', and 'postgresql+psycopg://'
    to 'postgresql+psycopg2://' while preserving all credentials, hostnames, ports,
    database paths, and query parameters.
    """
    if not url:
        return default
    if url.startswith('postgres://'):
        return 'postgresql+psycopg2://' + url[len('postgres://'):]
    if url.startswith('postgresql+psycopg://'):
        return 'postgresql+psycopg2://' + url[len('postgresql+psycopg://'):]
    if url.startswith('postgresql://'):
        return 'postgresql+psycopg2://' + url[len('postgresql://'):]
    return url


class _DevelopmentConfigMeta(type):
    @property
    def SECRET_KEY(cls):
        return os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

    @property
    def SQLALCHEMY_DATABASE_URI(cls):
        default_sqlite = 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'mindease.db')
        return _normalize_database_url(os.environ.get('DATABASE_URL'), default_sqlite)


class DevelopmentConfig(Config, metaclass=_DevelopmentConfigMeta):
    """Development configuration."""

    DEBUG = True

    @property
    def SQLALCHEMY_DATABASE_URI(self):
        default_sqlite = 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'mindease.db')
        return _normalize_database_url(os.environ.get('DATABASE_URL'), default_sqlite)

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

    @property
    def SQLALCHEMY_DATABASE_URI(cls):
        default_sqlite = 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'mindease.db')
        return _normalize_database_url(os.environ.get('DATABASE_URL'), default_sqlite)


class ProductionConfig(Config, metaclass=_ProductionConfigMeta):
    """Production configuration."""

    DEBUG = False
    SESSION_COOKIE_SECURE = True

    @property
    def SQLALCHEMY_DATABASE_URI(self):
        default_sqlite = 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'mindease.db')
        return _normalize_database_url(os.environ.get('DATABASE_URL'), default_sqlite)

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
