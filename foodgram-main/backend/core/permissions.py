from django.db.models import Model
from rest_framework.permissions import (
    SAFE_METHODS,
    BasePermission,
    IsAuthenticated
)
from rest_framework.request import Request
from rest_framework.viewsets import GenericViewSet

CHANGE_METHODS = ('PUT', 'PATCH', 'DELETE')


class ReadOnly(BasePermission):
    """Проверка на разрешение только редактирования."""

    def has_permission(self, request: Request, view: GenericViewSet) -> bool:
        return request.method in SAFE_METHODS


class IsAuthor(IsAuthenticated):
    """Проверка на доступность только автору."""

    def has_permission(self, request: Request, view: GenericViewSet) -> bool:
        return (
            request.method in CHANGE_METHODS
            and super().has_permission(request, view)
        )

    def has_object_permission(
        self, request: Request, view: GenericViewSet, obj: Model
    ) -> bool:
        return obj.author == request.user
