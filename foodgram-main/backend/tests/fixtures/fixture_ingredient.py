import pytest

from tests.utils.ingredient import INGREDIENT_DATA
from tests.utils.models import ingredient_model

Ingredient = ingredient_model()


@pytest.fixture
def ingredients():
    ingredients = [Ingredient(**item) for item in INGREDIENT_DATA]
    Ingredient.objects.bulk_create(ingredients)
    return list(Ingredient.objects.all())
