"""Django settings. The app is stateless: no database, sessions or auth."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "wewrite-dev-key-not-used-for-signing-anything")
DEBUG = os.environ.get("DJANGO_DEBUG") == "1"
ALLOWED_HOSTS = [".vercel.app", "localhost", "127.0.0.1", "testserver"] + [
    h for h in os.environ.get("EXTRA_HOSTS", "").split(",") if h
]

INSTALLED_APPS = ["fontgen"]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]
ROOT_URLCONF = "wewrite_site.urls"
WSGI_APPLICATION = "wewrite_site.wsgi.application"
DATABASES = {}
TEMPLATES = []
USE_TZ = True
DATA_UPLOAD_MAX_MEMORY_SIZE = 8 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 8 * 1024 * 1024
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
APPEND_SLASH = False
