from django.db.models import Model


def ingredient_model() -> Model:
    from api.models import Ingredient
    return Ingredient


def recipe_model() -> Model:
    from api.models import Recipe
    return Recipe


def recipe_favorite_model() -> Model:
    from api.models import RecipeFavorite
    return RecipeFavorite


def recipe_ingredients_model() -> Model:
    from api.models import RecipeIngredients
    return RecipeIngredients


def recipe_tags_model() -> Model:
    from api.models import RecipeTags
    return RecipeTags


def shopping_cart_model() -> Model:
    from api.models import ShoppingCart
    return ShoppingCart


def subscription_model() -> Model:
    from users.models import Subscription
    return Subscription


def tag_model() -> Model:
    from api.models import Tag
    return Tag
