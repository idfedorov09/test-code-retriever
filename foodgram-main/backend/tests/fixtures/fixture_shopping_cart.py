import pytest

from tests.utils.models import shopping_cart_model

ShoppingCart = shopping_cart_model()


@pytest.fixture
def three_shopping_cart(
    third_user, first_recipe, second_recipe, third_recipe
):
    shopping_cart = [
        ShoppingCart(author=third_user, recipe=recipe)
        for recipe in (first_recipe, second_recipe, third_recipe)
    ]
    ShoppingCart.objects.bulk_create(shopping_cart)
    return list(ShoppingCart.objects.all())


@pytest.fixture
def all_shopping_cart(third_user, all_recipes) -> list:
    shopping_cart = [
        ShoppingCart(author=third_user, recipe=recipe)
        for recipe in all_recipes
    ]
    ShoppingCart.objects.bulk_create(shopping_cart)
    return list(ShoppingCart.objects.all())
