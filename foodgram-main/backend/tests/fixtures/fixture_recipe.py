import pytest

from tests.utils.models import (
    recipe_ingredients_model,
    recipe_model,
    recipe_tags_model
)
from tests.utils.recipe import IMAGE

Recipe = recipe_model()
RecipeIngredients = recipe_ingredients_model()
RecipeTags = recipe_tags_model()


def create_recipe(data: dict):
    ingredients: list = data.pop('ingredients')
    tags: list = data.pop('tags')
    recipe = Recipe.objects.create(
        author=data['author'],
        name=data['name'],
        image=data['image'],
        text=data['text'],
        cooking_time=data['cooking_time']
    )

    recipe_ingredients = [
        RecipeIngredients(
            recipe=recipe,
            ingredient=ingredient['id'],
            amount=ingredient['amount']
        )
        for ingredient in ingredients
    ]
    RecipeIngredients.objects.bulk_create(recipe_ingredients)

    recipe_tags = [
        RecipeTags(
            recipe=recipe,
            tag=tag
        )
        for tag in tags
    ]
    RecipeTags.objects.bulk_create(recipe_tags)

    return recipe


@pytest.fixture
def first_recipe(ingredients, second_user, tags):
    return create_recipe({
        'author': second_user,
        'name': 'Нечто съедобное (это не точно)',
        'image': IMAGE,
        'text': 'Приготовьте как нибудь эти ингредиеты',
        'cooking_time': 5,
        'ingredients': [
            {'id': ingredients[0], 'amount': 10},
            {'id': ingredients[1], 'amount': 20},
        ],
        'tags': [tags[0], tags[1]]
    })


@pytest.fixture
def second_recipe(ingredients, second_user, tags):
    return create_recipe({
        'author': second_user,
        'name': 'Еще одна попытка приготовить еду',
        'image': IMAGE,
        'text': 'Вероятно стоит это смешать.',
        'cooking_time': 10,
        'ingredients': [
            {'id': ingredients[0], 'amount': 10},
            {'id': ingredients[1], 'amount': 20},
        ],
        'tags': [tags[1], tags[3]]
    })


@pytest.fixture
def third_recipe(ingredients, second_user, tags):
    return create_recipe({
        'author': second_user,
        'name': 'Еда без дополнительной обработки',
        'image': IMAGE,
        'text': 'Просто съесть',
        'cooking_time': 1,
        'ingredients': [
            {'id': ingredients[0], 'amount': 10}
        ],
        'tags': [tags[2]]
    })


@pytest.fixture
def fourth_recipe(ingredients, second_user, tags):
    return create_recipe({
        'author': second_user,
        'name': 'Нечто жареное',
        'image': IMAGE,
        'text': 'Жарить 10 минут',
        'cooking_time': 10,
        'ingredients': [
            {'id': ingredients[0], 'amount': 10}
        ],
        'tags': [tags[1]]
    })


@pytest.fixture
def fifth_recipe(ingredients, second_user, tags):
    return create_recipe({
        'author': second_user,
        'name': 'Варёное жареное',
        'image': IMAGE,
        'text': 'Варить 20 минут',
        'cooking_time': 25,
        'ingredients': [
            {'id': ingredients[1], 'amount': 20}
        ],
        'tags': [tags[2]]
    })


@pytest.fixture
def another_author_recipe(ingredients, first_user, tags):
    return create_recipe({
        'author': first_user,
        'name': 'Рецепт первого.',
        'image': IMAGE,
        'text': 'Варить 15 минут',
        'cooking_time': 15,
        'ingredients': [
            {'id': ingredients[0], 'amount': 15}
        ],
        'tags': [tags[0]]
    })


@pytest.fixture
def secound_author_recipes(
    first_recipe, second_recipe, third_recipe, fourth_recipe, fifth_recipe
) -> list:
    return [
        first_recipe, second_recipe, third_recipe, fourth_recipe, fifth_recipe
    ]


@pytest.fixture
def all_recipes(
    secound_author_recipes, another_author_recipe
) -> list:
    return [*secound_author_recipes, another_author_recipe]
