from django.conf import settings
from django.db.models.query import QuerySet
from rest_framework import serializers
from rest_framework.request import Request
from rest_framework.validators import UniqueTogetherValidator

from users.models import Subscription, User
from users.serializers.user import UserSerializer
from users.validators import SubscribeUniqueValidator


def get_base_recipe_serializer():
    """Отложенный импорт сериалайзера для избежания цикличного импорта."""

    from api.serializers import BaseRecipeSerializer
    return BaseRecipeSerializer


class SubscriptionGetSerializer(serializers.ModelSerializer):
    """Сериалайзер для фолловеров. Только для чтения."""

    is_subscribed = serializers.SerializerMethodField(
        method_name='get_is_subscribed'
    )
    recipes = serializers.SerializerMethodField(
        method_name='get_recipes'
    )
    recipes_count = serializers.IntegerField(
        source='recipes.count'
    )

    class Meta:
        model = User
        fields = (
            'email', 'id', 'username', 'first_name', 'last_name',
            'is_subscribed', 'recipes', 'recipes_count', 'avatar'
        )
        read_only_fields = fields

    def get_is_subscribed(self, obj: User):
        request: Request = self.context.get('request')
        if request is None or request.user.is_anonymous:
            return False
        return Subscription.objects.filter(
            user=request.user, author_recipe=obj
        ).exists()

    def get_recipes(self, obj: User):
        request: Request = self.context.get('request')
        recipes_limit: int = (
            request.GET.get('recipes_limit') if request
            else settings.RECIPES_LIMIT_MAX
        )
        queryset: QuerySet = obj.recipes.all()
        if recipes_limit:
            queryset = queryset[:int(recipes_limit)]
        serializer = get_base_recipe_serializer()
        return serializer(queryset, many=True).data


class SubscriptionChangedSerializer(serializers.ModelSerializer):
    """Сериалайзер для фолловеров. Только на запись."""

    author_recipe = UserSerializer
    user = UserSerializer

    class Meta:
        model = Subscription
        fields = ('author_recipe', 'user')
        validators = [
            UniqueTogetherValidator(
                queryset=Subscription.objects.all(),
                fields=('author_recipe', 'user'),
                message='Нельзя повторно подписаться на пользователя'
            ),
            SubscribeUniqueValidator(
                fields=('author_recipe', 'user')
            )
        ]

    def validate_subscription_exists(self):
        user = self.context.get('request').user
        author_recipe = self.validated_data.get('author_recipe')

        subscription = Subscription.objects.filter(
            author_recipe=author_recipe, user=user
        )
        if not subscription.exists():
            raise serializers.ValidationError(
                'У вас нет данного пользователя в подписчиках.'
            )

    def to_representation(self, instance):
        return SubscriptionGetSerializer(
            instance.author_recipe,
            context={'request': self.context.get('request')}
        ).data
