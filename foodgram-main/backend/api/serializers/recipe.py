from collections import OrderedDict
from functools import wraps
from typing import Optional

from django.db.models import Model
from rest_framework import serializers
from rest_framework.request import Request

from api.models import (
    Recipe,
    RecipeFavorite,
    RecipeIngredients,
    ShoppingCart,
    Tag
)
from api.serializers.recipe_ingredients import (
    RecipeIngredientsGetSerializer,
    RecipeIngredientsSetSerializer
)
from api.serializers.tag import TagSerializer
from core.serializers import Base64ImageField
from core.utils import many_unique_with_minimum_one_validate
from users.serializers import UserSerializer


class BaseRecipeSerializer(serializers.ModelSerializer):
    """
    Базовый сериалайзер рецептов.

    Содержит минимум необходимых полей для ответов на некоторые запросы.
    В перечень полей входят:
    - id
    - name
    - image
    - cooking_time
    """

    image = Base64ImageField()

    class Meta:
        model = Recipe
        fields = ('id', 'name', 'image', 'cooking_time')


class RecipeSerializer(BaseRecipeSerializer):
    """
    Сериалайзер-основа рецептов.

    Содержит дополнительные поля для сериалайзеров рецептов:
    теги, ингредиенты, автор и описание.
    """
    tags = TagSerializer(many=True)
    author = UserSerializer()
    ingredients = RecipeIngredientsGetSerializer(
        many=True,
        source='recipe_ingredients',
    )

    class Meta(BaseRecipeSerializer.Meta):
        abstract = True  # Чтоб лишний раз не искать "где использую"
        # Также хочу заметить, что в abstract.py не вынести, выйдет цикл
        fields = (
            *BaseRecipeSerializer.Meta.fields,
            'tags', 'ingredients', 'author', 'text'
        )


class RecipeGetSerializer(RecipeSerializer):
    """Сериалайзер рецептов для GET-методов и для ответов."""
    is_favorited = serializers.SerializerMethodField()
    is_in_shopping_cart = serializers.SerializerMethodField()

    class Meta(RecipeSerializer.Meta):
        fields = (
            *RecipeSerializer.Meta.fields,
            'is_favorited', 'is_in_shopping_cart'
        )
        read_only_fields = fields

    def get_is_exists(self, obj: Recipe, model: Model):
        request: Optional[Request] = self.context.get('request')
        if not request or request.user.is_anonymous:
            return False
        return model.objects.filter(
            author=request.user, recipe=obj
        ).exists()

    def get_is_favorited(self, obj: Recipe):
        return self.get_is_exists(obj, RecipeFavorite)

    def get_is_in_shopping_cart(self, obj: Recipe):
        return self.get_is_exists(obj, ShoppingCart)


class RecipeChangeSerializer(RecipeSerializer):
    """Сериалайзер для изменения рецептов"""

    tags = serializers.PrimaryKeyRelatedField(
        many=True, queryset=Tag.objects.all(),
        required=True
    )
    author = UserSerializer(default=serializers.CurrentUserDefault())
    ingredients = RecipeIngredientsSetSerializer(
        many=True,
        source='recipe_ingredients',
    )

    class Meta(RecipeSerializer.Meta):
        read_only_fields = ('author', )

    """
    Сообщение ревьюверу:
    Я пробовал делать через validate_<field>, но столкнулся с проблемой -
    мне приходилось делать дополнительные проверки внутри update-метода.
    https://gist.github.com/arefiture/bb78eac18e495ab6b8cec3db41c7d771
    Мне такой подход не понравился
    """
    def validate(self, data: OrderedDict):
        ingredients = data.get('recipe_ingredients')
        many_unique_with_minimum_one_validate(
            data_list=ingredients, field_name='ingredients',
            singular='ингредиент', plural='ингредиенты'
        )

        tags = data.get('tags')
        many_unique_with_minimum_one_validate(
            data_list=tags, field_name='tags',
            singular='тег', plural='теги'
        )

        return data

    def added_tags_ingredients(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # У create метода первый - validated_data
            # У update метода сначала instance, потом validated_data
            # Т.е. для них последний эл - validated_data
            validated_data: dict = args[-1]
            ingredients: list[dict] = validated_data.pop('recipe_ingredients')
            tags = validated_data.pop('tags')
            recipe: Recipe = func(self, *args, **kwargs)

            recipe.tags.set(tags)
            ingredient_recipe = [
                RecipeIngredients(
                    recipe=recipe,
                    ingredient=ingredient.get('id'),
                    amount=ingredient.get('amount')
                ) for ingredient in ingredients
            ]
            RecipeIngredients.objects.bulk_create(ingredient_recipe)
            return recipe
        return wrapper

    @added_tags_ingredients
    def create(self, validated_data: dict):
        return Recipe.objects.create(**validated_data)

    @added_tags_ingredients
    def update(self, instance: Recipe, validated_data: dict):
        super().update(instance, validated_data)
        instance.recipe_ingredients.all().delete()
        return instance

    def to_representation(self, instance):
        return RecipeGetSerializer(
            instance, context=self.context
        ).data
