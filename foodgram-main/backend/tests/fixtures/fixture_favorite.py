import pytest

from tests.utils.models import recipe_favorite_model

RecipeFavorite = recipe_favorite_model()


@pytest.fixture
def all_favorite(third_user, all_recipes):
    favorites = [
        RecipeFavorite(author=third_user, recipe=recipe)
        for recipe in all_recipes
    ]
    RecipeFavorite.objects.bulk_create(favorites)
    return list(RecipeFavorite.objects.all())
