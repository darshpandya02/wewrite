from django.conf import settings
from django.urls import path, re_path
from django.views.static import serve

from fontgen import views

urlpatterns = [
    path("api/health", views.health),
    path("api/style", views.style),
    path("api/generate", views.generate),
    path("api/font", views.font),
    path("api/vectorize", views.vectorize_view),
]

if settings.DEBUG:  # local development only; on Vercel the CDN serves public/
    public = settings.BASE_DIR / "public"
    urlpatterns += [
        path("", serve, {"path": "index.html", "document_root": public}),
        re_path(r"^(?P<path>.+)$", serve, {"document_root": public}),
    ]

