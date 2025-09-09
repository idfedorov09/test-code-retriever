from tests.utils.recipe import RESPONSE_SCHEMA_SHORT_RECIPE
from tests.utils.user import URL_CREATE_USER, URL_GET_USER

# Адреса страниц
URL_CREATE_SUBSCRIBE = URL_GET_USER + 'subscribe/'
URL_GET_SUBSCRIPTIONS = URL_CREATE_USER + 'subscriptions/'

# Схемы валидации данных в ответах методов
RESPONSE_SCHEMA_SUBSCRIPTION = {
    "type": "object",
    "properties": {
        "id": {"type": "number"},
        "username": {"type": "string"},
        "first_name": {"type": "string"},
        "last_name": {"type": "string"},
        "email": {"type": "string"},
        "is_subscribed": {"type": "boolean"},
        "avatar": {"type": ["string", "null"]},
        "recipes_count": {"type": "number"},
        "recipes": {
            "type": "array",
            "items": RESPONSE_SCHEMA_SHORT_RECIPE
        }
    },
    "required": [
        "id", "username", "first_name", "last_name", "email",
        "is_subscribed", "recipes", "recipes_count", "avatar"
    ],
    "additionalProperties": False
}

RESPONSE_SCHEMA_SUBSCRIPTIONS = {
    "type": "object",
    "properties": {
        "count": {"type": "number"},
        "next": {"type": ["string", "null"]},
        "previous": {"type": ["string", "null"]},
        "results": {
            "type": "array",
            "items": RESPONSE_SCHEMA_SUBSCRIPTION
        }
    },
    "required": ["count", "next", "previous", "results"],
    "additionalProperties": False
}
