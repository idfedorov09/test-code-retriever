from django.shortcuts import get_object_or_404, redirect
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.serializers import Serializer
from rest_framework.views import APIView

from api.filters import RecipeFilter
from api.models import Recipe
from api.serializers import RecipeChangeSerializer, RecipeGetSerializer
from api.views.recipe_favorite import RecipeFavoriteMixin
from api.views.shopping_cart import ShoppingCartMixin
from core.permissions import IsAuthor, ReadOnly


class RecipeViewSet(
    viewsets.ModelViewSet,
    RecipeFavoriteMixin,
    ShoppingCartMixin
):
    """Вьюсет для рецептов и связных по /recipes/ действий."""

    queryset = Recipe.objects.all()
    permission_classes = [ReadOnly]
    filter_backends = [DjangoFilterBackend]
    filterset_class = RecipeFilter
    serializer_class = RecipeChangeSerializer
    ordering = ['-pub_date']

    def get_permissions(self):
        if self.action == 'download_shopping_cart':
            self.permission_classes = [IsAuthenticated]
        elif self.request.method in ("GET", "POST"):
            self.permission_classes = [IsAuthenticated | ReadOnly]
        elif self.request.method in ("PATCH", "DELETE"):
            self.permission_classes = [IsAuthor]
        return super().get_permissions()

    def get_serializer_class(self):
        if self.action in ('list', 'retrieve'):
            return RecipeGetSerializer
        elif self.action in ['create', 'update', 'partial_update']:
            return RecipeChangeSerializer
        return super().get_serializer_class()

    def perform_create(self, serializer: Serializer):
        serializer.is_valid(raise_exception=True)
        serializer.save(author=self.request.user)

    def partial_update(self, request: Request, *args, **kwargs):
        instance: Recipe = self.get_object()
        data = request.data.copy()

        serializer: Serializer = self.get_serializer(
            instance, data=data, partial=True
        )
        serializer.is_valid(raise_exception=True)
        super().perform_update(serializer)
        return Response(serializer.data)

    @action(detail=True, methods=['GET'], url_path='get-link')
    def get_short_link(self, request: Request, pk: int):
        try:
            recipe: Recipe = self.get_object()
        except Recipe.DoesNotExist:
            return Response(
                {'message': 'Не существует такой записи'},
                status=status.HTTP_404_NOT_FOUND
            )

        scheme = request.scheme
        host = request.get_host()
        domain = f'{scheme}: //{host}'
        return Response(
            {'short-link': f'{domain}/s/{recipe.short_link}'},
            status=status.HTTP_200_OK
        )


class RecipeRedirectView(APIView):
    """
    Вьюсет редиректа с короткой ссылки рецепта на абсолютный адрес.

    Важно: редирект идёт на адрес фронта, а не /api/!
    """

    permission_classes = [ReadOnly]

    def get(self, request: Request, short_link: str):
        recipe = get_object_or_404(Recipe, short_link=short_link)
        return redirect(recipe.get_frontend_absolute_url())
