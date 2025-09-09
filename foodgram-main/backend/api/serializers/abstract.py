"""
Для ревьювера: Я знаю, что у <ClassSerializer>.Meta нет
атрибута abstract. Само слово abstract в данном случае
несёт поясняющий характер, не более.
"""

from rest_framework import serializers
from rest_framework.validators import UniqueTogetherValidator

from api.models.abstract_models import BaseActionRecipeModel
from api.models.recipe import Recipe
from api.serializers.recipe import BaseRecipeSerializer
from users.serializers import UserSerializer


class BaseRecipeActionSerializer(serializers.ModelSerializer):
    """
    Сериалайзер для добавления рецептов и авторов в связанные таблицы.

    Используется при добавлении в корзину, в избранные.
    """

    author = UserSerializer
    recipe = serializers.PrimaryKeyRelatedField(
        queryset=Recipe.objects.all()
    )

    class Meta:
        model = None  # Назначаем у дочерних классов
        fields = ('author', 'recipe')
        error_message = ''

    @classmethod
    def get_validators(cls):
        return [
            UniqueTogetherValidator(
                queryset=cls.Meta.model.objects.all(),
                fields=('author', 'recipe'),
                message=cls.Meta.error_message
            )
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Meta.validators = self.get_validators()

    def to_representation(self, instance: BaseActionRecipeModel):
        return BaseRecipeSerializer(instance.recipe).data
