import pytest
from rest_framework.test import APIClient

from tests.base_test import BaseTest
from tests.utils.general import NOT_EXISTING_ID
from tests.utils.tag import (
    DENY_CHANGE_METHOD,
    RESPONSE_SCHEMA_TAG,
    RESPONSE_SCHEMA_TAGS,
    URL_GET_TAG,
    URL_TAGS
)


@pytest.mark.django_db(transaction=True)
class TestTags(BaseTest):

    @pytest.mark.parametrize(
        'method, method_info', DENY_CHANGE_METHOD.items()
    )
    @pytest.mark.usefixtures('tags')
    def test_bad_request_methods(
        self, first_user_authorized_client: APIClient, method: str,
        method_info: dict
    ):
        self.url_is_not_allowed_for_method(
            client=first_user_authorized_client,
            url=method_info['url'],
            method=method
        )

    @pytest.mark.parametrize(
        'client',
        ['api_client', 'first_user_authorized_client'],
        indirect=True
    )
    @pytest.mark.usefixtures('tags')
    def test_get_tags(self, client: APIClient):
        self.url_get_resource(
            client=client,
            url=URL_TAGS,
            response_schema=RESPONSE_SCHEMA_TAGS
        )

    @pytest.mark.parametrize(
        'client',
        ['api_client', 'first_user_authorized_client'],
        indirect=True
    )
    def test_get_tag_detail(self, client: APIClient, tags: list):
        self.url_get_resource(
            client=client,
            url=URL_GET_TAG.format(id=tags[0].id),
            response_schema=RESPONSE_SCHEMA_TAG
        )

    @pytest.mark.usefixtures('tags')
    def test_non_existing_tag(self, first_user_authorized_client: APIClient):
        self.url_is_missing_for_method(
            client=first_user_authorized_client,
            url=URL_GET_TAG.format(id=NOT_EXISTING_ID),
            method='get'
        )
