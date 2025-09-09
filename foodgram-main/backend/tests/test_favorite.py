import pytest
from django.db.models import Model
from rest_framework.test import APIClient

from tests.base_test import BaseTest
from tests.utils.favorite import URL_FAVORITE
from tests.utils.general import NOT_EXISTING_ID
from tests.utils.models import recipe_favorite_model
from tests.utils.recipe import RESPONSE_SCHEMA_SHORT_RECIPE

RecipeFavorite = recipe_favorite_model()


@pytest.mark.django_db(transaction=True)
class TestFavorite(BaseTest):

    def test_add_to_favorite_unauthorized(
        self, api_client: APIClient, second_recipe: Model
    ):
        self.url_requires_authorization(
            client=api_client,
            url=URL_FAVORITE.format(id=second_recipe.id)
        )

    @pytest.mark.usefixtures('all_favorite')
    def test_add_again_to_favorite(
        self, third_user_authorized_client: APIClient, third_recipe: Model
    ):
        self.check_db_no_changes_made(
            client=third_user_authorized_client,
            url=URL_FAVORITE.format(id=third_recipe.id),
            model=RecipeFavorite
        )

    def test_add_non_existing_recipe_to_favorite(
        self, third_user_authorized_client: APIClient
    ):
        self.url_is_missing_for_method(
            client=third_user_authorized_client,
            url=URL_FAVORITE.format(id=NOT_EXISTING_ID),
            method='post'
        )

    def test_add_to_favorite_authorized(
        self, third_user_authorized_client: APIClient, third_user: Model,
        first_recipe: Model
    ):
        self.url_creates_resource(
            client=third_user_authorized_client,
            url=URL_FAVORITE.format(id=first_recipe.id),
            model=RecipeFavorite,
            filters={'author': third_user, 'recipe': first_recipe},
            response_schema=RESPONSE_SCHEMA_SHORT_RECIPE
        )

    def test_delete_favorite_unauthorized(
        self, api_client: APIClient, all_favorite: list
    ):
        favorite = all_favorite[0]
        self.url_requires_authorization(
            client=api_client,
            url=URL_FAVORITE.format(id=favorite.recipe_id),
            method='delete'
        )

    def test_delete_not_added_from_favorite(
        self, second_user_authorized_client: APIClient, all_favorite: list
    ):
        favorite = all_favorite[0]
        self.url_bad_request_for_invalid_data(
            client=second_user_authorized_client,
            url=URL_FAVORITE.format(id=favorite.recipe_id),
            method='delete'
        )

    def test_delete_non_existing_recipe_from_favorite(
        self, third_user_authorized_client: APIClient
    ):
        self.url_is_missing_for_method(
            client=third_user_authorized_client,
            url=URL_FAVORITE.format(id=NOT_EXISTING_ID),
            method='delete'
        )

    def test_delete_favorite(
        self, third_user_authorized_client: APIClient, all_favorite: list
    ):
        favorite = all_favorite[0]
        self.url_delete_resouce(
            client=third_user_authorized_client,
            url=URL_FAVORITE.format(id=favorite.recipe_id),
            model=RecipeFavorite,
            item_id=favorite.id
        )
