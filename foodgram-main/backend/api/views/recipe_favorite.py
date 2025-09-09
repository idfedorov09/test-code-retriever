from django.shortcuts import get_object_or_404
from rest_framework.decorators import action
from rest_framework.request import Request

from api.models import Recipe, RecipeFavorite
from api.serializers import RecipeFavoriteSerializer
from core.utils import object_delete, object_update


class RecipeFavoriteMixin:
    """Отдельный блок действий для управления избранными рецептами."""

    def get_data(self, request: Request, pk: int) -> dict:
        return {
            'author': request.user,
            'recipe': get_object_or_404(Recipe, id=pk)
        }

    @action(detail=True, methods=['POST'], url_path='favorite')
    def post_favorite(self, request: Request, pk: int):
        data: dict = self.get_data(request=request, pk=pk)
        serializer = RecipeFavoriteSerializer(
            data={key: obj.id for key, obj in data.items()},
            context={'request': request}
        )
        return object_update(serializer=serializer)

    @post_favorite.mapping.delete
    def delete_favorite(self, request: Request, pk: int):
        return object_delete(
            data=self.get_data(request=request, pk=pk),
            error_mesage='У вас нет данного рецепта в избранном.',
            model=RecipeFavorite
        )
