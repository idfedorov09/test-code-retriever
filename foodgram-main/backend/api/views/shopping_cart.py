import csv
import io
from datetime import datetime

from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework.decorators import action
from rest_framework.request import Request

from api.models import Recipe, ShoppingCart
from api.serializers import (
    DownloadShoppingCartSerializer,
    ShoppingCartSerializer
)
from core.utils import object_delete, object_update


class ShoppingCartMixin:
    """Отдельный блок действий для управления корзиной с рецептами."""

    def get_data(self, request: Request, pk: int) -> dict:
        return {
            'author': request.user,
            'recipe': get_object_or_404(Recipe, id=pk)
        }

    @action(detail=True, methods=['POST'], url_path='shopping_cart')
    def post_shopping_cart(self, request: Request, pk: int):
        data: dict = self.get_data(request=request, pk=pk)
        serializer = ShoppingCartSerializer(
            data={key: obj.id for key, obj in data.items()},
            context={'request': request}
        )
        return object_update(serializer=serializer)

    @post_shopping_cart.mapping.delete
    def delete_shopping_cart(self, request: Request, pk: int):
        return object_delete(
            data=self.get_data(request=request, pk=pk),
            error_mesage='У вас нет данного рецепта в корзине.',
            model=ShoppingCart
        )

    @action(detail=False, methods=['GET'], url_path='download_shopping_cart')
    def download_shopping_cart(self, request):
        serializer = DownloadShoppingCartSerializer(
            instance=ShoppingCart.objects.all(),
            many=True, context={'request': request}
        )

        now = datetime.now()
        formatted_time = now.strftime('%d-%m-%Y_%H_%M_%S')

        response = HttpResponse(content_type='text/csv; charset=cp1251')
        filename = f'shopping_cart.csv_{request.user.id}_{formatted_time}'
        response['Content-Disposition'] = (
            f'attachment filename="{filename}"'
        )

        csv_buffer = io.TextIOWrapper(response, encoding='cp1251', newline='')
        writer = csv.writer(csv_buffer)
        writer.writerow(['Ингредиент', 'Единица измерения', 'Количество'])
        if serializer.data:
            ingredients = serializer.data[0]['ingredients']

            rows = map(
                lambda ingredient: [
                    ingredient['name'],
                    ingredient['measurement_unit'],
                    ingredient['total_amount']
                ], ingredients
            )
            writer.writerows(rows)

        csv_buffer.flush()  # Сбрасываем буфер, чтобы данные записались
        return response
