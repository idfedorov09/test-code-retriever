from django.shortcuts import get_object_or_404
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request

from core.utils import object_delete, object_update
from users.models import Subscription, User
from users.serializers import (
    SubscriptionChangedSerializer,
    SubscriptionGetSerializer
)


class SubscriptionMixin:
    """Отдельный блок действий для управления подписчиками."""

    def get_data(self, request: Request, id: int) -> dict:
        return {
            'user': request.user,
            'author_recipe': get_object_or_404(User, id=id)
        }

    @action(['GET'], detail=False, url_path='subscriptions')
    def subscriptions(self, request: Request):
        user = request.user
        queryset = User.objects.filter(authors__user=user)
        pages = self.paginate_queryset(queryset)

        serializer = SubscriptionGetSerializer(
            pages,
            many=True,
            context={'request': request}
        )
        return self.get_paginated_response(serializer.data)

    @action(
        detail=True,
        methods=['POST'],
        url_path='subscribe',
        permission_classes=[IsAuthenticated]
    )
    def post_subscribe(self, request: Request, id: int):
        data: dict = self.get_data(request=request, id=id)
        serializer = SubscriptionChangedSerializer(
            data={key: obj.id for key, obj in data.items()},
            context={'request': request}
        )
        return object_update(serializer=serializer)

    @post_subscribe.mapping.delete
    def delete_subscribe(self, request: Request, id: int):
        return object_delete(
            data=self.get_data(request=request, id=id),
            error_mesage='У вас нет данного пользователя в подписчиках.',
            model=Subscription
        )
