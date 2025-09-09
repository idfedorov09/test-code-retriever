import re

import pytest
from django.db.models import Model, Q
from pytest_django.fixtures import SettingsWrapper
from pytest_lazyfixture import lazy_fixture
from rest_framework.response import Response
from rest_framework.test import APIClient

from tests.base_test import BaseTest
from tests.utils.general import NOT_EXISTING_ID
from tests.utils.models import (
    recipe_favorite_model,
    recipe_ingredients_model,
    recipe_model,
    recipe_tags_model,
    shopping_cart_model
)
from tests.utils.recipe import (
    BODY_ONLY_POST_BAD_REQUEST,
    BODY_POST_AND_PATH_BAD_REQUESTS,
    BODY_UPDATE_VALID,
    IMAGE,
    RESPONSE_SCHEMA_RECIPE,
    RESPONSE_SCHEMA_RECIPES,
    RESPONSE_SCHEMA_SHORT_LINK,
    URL_GET_FRONT_RECIPE,
    URL_GET_RECIPE,
    URL_GET_SHORT_LINK,
    URL_RECIPES,
    URL_SHORT_LINK
)
from tests.utils.tag import TAG_SET_SLUGS

Recipe = recipe_model()
RecipeFavorite = recipe_favorite_model()
RecipeIngredients = recipe_ingredients_model()
RecipeTags = recipe_tags_model()
ShoppingCart = shopping_cart_model()


@pytest.mark.django_db(transaction=True)
class TestRecipe(BaseTest):

    @pytest.mark.parametrize(
        'subname_test, body', (
            BODY_ONLY_POST_BAD_REQUEST | BODY_POST_AND_PATH_BAD_REQUESTS
        ).items()
    )
    @pytest.mark.usefixtures('subname_test')
    def test_bad_request_method_post(
        self, second_user_authorized_client: APIClient, body: dict
    ):
        self.url_bad_request_for_invalid_data(
            client=second_user_authorized_client,
            url=URL_RECIPES,
            data=body
        )

    @pytest.mark.parametrize(
        'subname_test, body', BODY_POST_AND_PATH_BAD_REQUESTS.items()
    )
    @pytest.mark.usefixtures('subname_test')
    def test_bad_request_method_patch(
        self, second_user_authorized_client: APIClient, first_recipe: Model,
        body: dict
    ):
        self.url_bad_request_for_invalid_data(
            client=second_user_authorized_client,
            url=URL_GET_RECIPE.format(id=first_recipe.id),
            method='patch',
            data=body
        )

    @pytest.mark.usefixtures('secound_author_recipes')
    def test_add_recipe_unauthorized(self, api_client: APIClient):
        self.url_requires_authorization(
            client=api_client,
            url=URL_RECIPES
        )

    def test_update_recipe_unauthorized(
        self, api_client: APIClient, first_recipe: Model
    ):
        self.url_requires_authorization(
            client=api_client,
            url=URL_GET_RECIPE.format(id=first_recipe.id),
            method='patch',
            data=BODY_UPDATE_VALID
        )

    def test_update_recipe_non_author(
        self, first_user_authorized_client: APIClient, first_recipe: Model
    ):
        self.url_access_denied(
            client=first_user_authorized_client,
            url=URL_GET_RECIPE.format(id=first_recipe.id),
            method='patch',
            body=BODY_UPDATE_VALID
        )

    def test_update_recipe_none_existing_recipe(
        self, second_user_authorized_client: APIClient
    ):
        self.url_is_missing_for_method(
            client=second_user_authorized_client,
            url=URL_GET_RECIPE.format(id=NOT_EXISTING_ID),
            method='patch'
        )

    @pytest.mark.usefixtures('second_user')
    def test_add_recipe_authorized(  # TODO: Возможно, вынести часть логики
        self, second_user_authorized_client: APIClient, ingredients: list,
        tags: list
    ):
        count_recipe_ingredients = RecipeIngredients.objects.count()
        count_recipe_tags = RecipeTags.objects.count()

        body = {
            'ingredients': [
                {'id': ingredient.id, 'amount': 10 + ingredient.id}
                for index, ingredient in enumerate(ingredients)
                if index in (1, 2)
            ],
            'tags': [
                tag.id for index, tag in enumerate(tags) if index in (1, 2)
            ],
            'image': IMAGE,
            'name': 'Новый рецепт',
            'text': 'Забытый рецепт с неизвестными ингредиентами.',
            'cooking_time': 5
        }
        count_body_recipe_ingredients = len(body.get('ingredients'))
        count_body_recipe_tags = len(body.get('tags'))

        response: Response = self.url_creates_resource(
            client=second_user_authorized_client,
            url=URL_RECIPES,
            data=body,
            model=Recipe,
            filters={
                'name': body.get('name'),
                'text': body.get('text'),
                'cooking_time': body.get('cooking_time')
            },
            response_schema=RESPONSE_SCHEMA_RECIPE
        )

        data: dict = response.json()
        id_new_recipe: int = data['id']

        new_recipe_tags = RecipeTags.objects.filter(
            recipe_id=id_new_recipe,
            tag_id__in=body.get('tags')
        )
        assert new_recipe_tags.exists() and new_recipe_tags.count() == (
            count_recipe_tags + count_body_recipe_tags
        ), (
            'Убедитесь, что в связующую рецепты+теги добавились записи '
            'в БД с ожидаемыми полями.'
        )
        # Для каждого генерим Q-Объект из условия ингры+количества
        query = Q()
        for data in body.get('ingredients'):
            query |= Q(
                recipe_id=id_new_recipe,
                ingredient_id=data['id'],
                amount=data['amount']
            )
        new_recipe_ingredients = RecipeIngredients.objects.filter(query)
        assert (
            new_recipe_ingredients.exists()
            and new_recipe_ingredients.count() == (
                count_recipe_ingredients + count_body_recipe_ingredients
            )
        ), (
            'Убедитесь, что в связующую рецепты+ингредиенты добавились записи '
            'в БД с ожидаемыми полями.'
        )

    @pytest.mark.parametrize(
        'client',
        ['api_client', 'first_user_authorized_client'],
        indirect=True
    )
    @pytest.mark.usefixtures('all_recipes')
    def test_get_recipes(
        self, client: APIClient, settings: SettingsWrapper
    ):
        response: Response = client.get(URL_RECIPES)
        self.url_get_resource(
            response=response,
            url=URL_RECIPES,
            response_schema=RESPONSE_SCHEMA_RECIPES
        )
        # TODO: если вдруг падают тесты, нужно использовать проверку:
        # page_size = YourViewSet().pagination_class.page_size
        self.url_pagination_results(
            data=response.json(),
            limit=settings.REST_FRAMEWORK.get("PAGE_SIZE", 10)
        )

    @pytest.mark.parametrize('count', [1, 5, 20])
    @pytest.mark.usefixtures('secound_author_recipes')
    def test_get_recipes_paginated(
        self, api_client: APIClient, count: int
    ):
        url = URL_RECIPES + '?limit=' + str(count)
        response: Response = api_client.get(url)
        self.url_get_resource(
            response=response,
            url=url,
            response_schema=RESPONSE_SCHEMA_RECIPES
        )
        self.url_pagination_results(
            data=response.json(),
            limit=count
        )

    @pytest.mark.parametrize(
        'user', [
            lazy_fixture('first_user'),
            lazy_fixture('second_user'),
            lazy_fixture('third_user')
        ]
    )
    @pytest.mark.usefixtures('all_recipes')
    def test_get_recipes_filter_by_author(
        self, api_client: APIClient, settings: SettingsWrapper, user: Model
    ):
        url = URL_RECIPES + '?author=' + str(user.id)
        response: Response = api_client.get(url)
        self.url_get_resource(
            response=response,
            url=url,
            response_schema=RESPONSE_SCHEMA_RECIPES
        )
        self.url_filters_by_query_parameters(
            response=response,
            model=Recipe,
            filters={'author_id': user.id},
            limit=settings.REST_FRAMEWORK.get("PAGE_SIZE", 10)
        )

    @pytest.mark.parametrize(
        'count_tags, slug_list', TAG_SET_SLUGS.items()
    )
    @pytest.mark.usefixtures('all_recipes', 'count_tags')
    def test_get_recipes_filter_by_tags(
        self, api_client: APIClient, settings: SettingsWrapper, slug_list: list
    ):
        url = URL_RECIPES + '?' + '&'.join(
            [f'tags={slug}' for slug in slug_list]
        )
        response: Response = api_client.get(url)
        self.url_get_resource(
            response=response,
            url=url,
            response_schema=RESPONSE_SCHEMA_RECIPES
        )
        self.url_filters_by_query_parameters(
            response=response,
            model=Recipe,
            filters={'tags__slug__in': slug_list},
            limit=settings.REST_FRAMEWORK.get("PAGE_SIZE", 10)
        )

    @pytest.mark.parametrize(
        'client', [
            lazy_fixture('api_client'),
            lazy_fixture('second_user_authorized_client')
        ]
    )
    def test_get_recipe_detail(
        self, client: APIClient, first_recipe: Model
    ):
        self.url_get_resource(
            client=client,
            url=URL_GET_RECIPE.format(id=first_recipe.id),
            response_schema=RESPONSE_SCHEMA_RECIPE
        )

    @pytest.mark.parametrize(
        'client', [
            lazy_fixture('api_client'),
            lazy_fixture('second_user_authorized_client')
        ]
    )
    @pytest.mark.parametrize(
        'recipe', [
            lazy_fixture('first_recipe'),
            lazy_fixture('third_recipe'),
            lazy_fixture('fifth_recipe')
        ]
    )
    def test_get_short_link(
        self, client: APIClient, recipe: Model
    ):
        url = URL_GET_SHORT_LINK.format(id=recipe.id)
        response: Response = client.get(url)
        self.url_get_resource(
            response=response,
            url=url,
            response_schema=RESPONSE_SCHEMA_SHORT_LINK
        )
        response_json: dict = response.json()
        short_link: str = response_json.get('short-link', '')
        pattern = r'^.*/s/[a-zA-Z0-9]{2,6}$'
        assert re.match(pattern, short_link) is not None, (
            'Убедитесь, что в ответе запроса на адрес `{url}` значение '
            '`short_link` соответствует регулярному выражению `{pattern}`.'
        )

    @pytest.mark.parametrize(
        'client', [
            lazy_fixture('api_client'),
            lazy_fixture('second_user_authorized_client')
        ]
    )
    @pytest.mark.parametrize(
        'recipe', [
            lazy_fixture('second_recipe'),
            lazy_fixture('fourth_recipe'),
            lazy_fixture('another_author_recipe')
        ]
    )
    def test_redirect_short_link(
        self, client: APIClient, recipe: Model
    ):
        self.url_redirects_with_found_status(
            client=client,
            url=URL_SHORT_LINK.format(uuid=recipe.short_link),
            expected_redirect_url=URL_GET_FRONT_RECIPE.format(id=recipe.id)
        )

    def test_update_recipe_author(  # TODO: Возможно, вынести часть логики
        self, second_user_authorized_client: APIClient, first_recipe: Model,
        tags: list, ingredients: list
    ):
        url = URL_GET_RECIPE.format(id=first_recipe.id)
        # Из-за боли с транзакциями делаем немного махинаций
        body = BODY_UPDATE_VALID.copy()
        body['tags'] = [
            tags[id].id for id in body['tags']
        ]
        for ingredient in body['ingredients']:
            ingredient['id'] = ingredients[ingredient['id']].id

        response: Response = second_user_authorized_client.patch(url, body)

        self.url_get_resource(
            response=response,
            url=url,
            response_schema=RESPONSE_SCHEMA_RECIPE
        )

        recipes = Recipe.objects.filter(
            name=body.get('name'),
            text=body.get('text'),
            cooking_time=body.get('cooking_time'))
        assert recipes.exists(), (
            'Убедитесь, что поля в рецепте обновились.'
        )
        id_recipe = recipes[0].id
        recipe_tags = RecipeTags.objects.filter(
            recipe_id=id_recipe,
            tag_id__in=body.get('tags')
        )
        assert (
            recipe_tags.exists()
            and recipe_tags.count() == len(body.get('tags'))
        ), (
            'Убедитесь, что в связующей рецепты+теги только обновленные записи'
        )
        query = Q()
        for data in body.get('ingredients'):
            query |= Q(
                recipe_id=id_recipe,
                ingredient_id=data['id'],
                amount=data['amount']
            )
        recipe_ingredients = RecipeIngredients.objects.filter(query)
        assert (
            recipe_ingredients.exists()
            and recipe_ingredients.count() == len(
                body.get('ingredients')
            )
        ), (
            'Убедитесь, что в связующей рецепты+ингредиенты только '
            'обновленные записи'
        )

    @pytest.mark.parametrize(
        'user_status, client',
        [
            ('anonymous', lazy_fixture('api_client')),
            ('authorized', lazy_fixture('third_user_authorized_client'))
        ]
    )
    @pytest.mark.parametrize(
        'flag', [0, 1]
    )
    @pytest.mark.parametrize(
        'Model, flag_name',
        [
            (ShoppingCart, 'is_in_shopping_cart'),
            (RecipeFavorite, 'is_favorited')
        ]
    )
    def test_get_recipes_with_flag(
        self, client: APIClient, third_user: Model, flag: int,
        user_status: str, Model: Model, flag_name: str
    ):
        url = URL_RECIPES + f'?{flag_name}={flag}'
        response: Response = client.get(url)
        self.url_get_resource(
            response=response,
            url=url,
            response_schema=RESPONSE_SCHEMA_RECIPES
        )
        response_json: dict = response.json()
        response_count: int = response_json['count']
        if flag == 1 and user_status == 'anonymous':
            response_results = response_json['results']
            assert response_count == 0 == len(response_results), (
                'Убедитесь, что для неавторизованного вернётся пустой список.'
            )
        else:
            recipe_in_response = {
                recipe['id'] for recipe in response_json['results']
            }
            if user_status == 'anonymous':
                recipe_in_DB = {
                    cart.id for cart in Recipe.objects.all()
                }
                assert (
                    len(recipe_in_DB) == response_count
                    and recipe_in_response == recipe_in_DB
                ), (
                    'Для неавторизованного пользователя должны вернуться '
                    'все рецепты'
                )
            else:
                recipe_in_DB = {
                    cart.id for cart in Model.objects.filter(
                        author=third_user
                    )
                }
                if flag:
                    assert (
                        response_count == len(recipe_in_response)
                        and recipe_in_response == recipe_in_DB
                    ), (
                        f'Убедитесь, что фильтр `{flag_name}` на '
                        'странице рецептов работает.'
                    )
                else:
                    assert (
                        response_count == len(recipe_in_response)
                        and len(
                            recipe_in_response.intersection(recipe_in_DB)
                        ) == 0
                    ), (
                        f'Убедитесь, что фильтр `{flag_name}` на '
                        'странице рецептов работает.'
                    )

    def test_delete_recipe_unauthorized(
        self, api_client: APIClient, first_recipe: Model
    ):
        id_recipe: int = first_recipe.id
        self.url_requires_authorization(
            client=api_client,
            url=URL_GET_RECIPE.format(id=id_recipe),
            method='delete'
        )

    def test_delete_not_added_from_recipe(
        self, first_user_authorized_client: APIClient, first_recipe: Model
    ):
        id_recipe: int = first_recipe.id
        self.url_access_denied(
            client=first_user_authorized_client,
            url=URL_GET_RECIPE.format(id=id_recipe),
            method='delete'
        )

    def test_delete_non_existing_recipe(
        self, second_user_authorized_client: APIClient
    ):
        self.url_is_missing_for_method(
            client=second_user_authorized_client,
            url=URL_GET_RECIPE.format(id=NOT_EXISTING_ID),
            method='delete'
        )

    def test_delete_recipe(
        self, second_user_authorized_client: APIClient, first_recipe: Model
    ):
        id_recipe: int = first_recipe.id
        self.url_delete_resouce(
            client=second_user_authorized_client,
            url=URL_GET_RECIPE.format(id=id_recipe),
            model=Recipe,
            item_id=id_recipe
        )
        assert not RecipeTags.objects.filter(recipe_id=first_recipe.id), (
            'Убедитесь, что удалились связанные с рецептом теги.'
        )
        assert (
            not RecipeIngredients.objects.filter(recipe_id=first_recipe.id)
        ), (
            'Убедитесь, что удалились связанные с рецептом ингредиенты.'
        )
