from djoser.serializers import UserSerializer as DjoserUserSerializer
from rest_framework import serializers

from core.serializers import Base64ImageField
from users.models import Subscription, User


class AvatarSerializer(serializers.Serializer):
    """Сериалайзер под аватар."""

    avatar = Base64ImageField(required=False)


class CurrentUserSerializer(DjoserUserSerializer):
    """Сериалайзер под текущего пользователя (для /me)."""

    is_subscribed = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = (
            'id',
            'email',
            'username',
            'first_name',
            'last_name',
            'avatar',
            'is_subscribed',
        )

    def get_is_subscribed(self, obj):
        return False


class UserSerializer(CurrentUserSerializer):
    """Общий сериалайзер пользователя."""

    def get_is_subscribed(self, obj):
        request = self.context.get('request')
        return Subscription.objects.filter(
            user_id=request.user.id,
            author_recipe_id=obj.id
        ).exists()
