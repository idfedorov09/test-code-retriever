from tests.utils.tag import RESPONSE_SCHEMA_TAG
from tests.utils.user import RESPONSE_SCHEMA_USER

# Константы из postman
IMAGE = (
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAgMAAABieywa'
    'AAAACVBMVEUAAAD///9fX1/S0ecCAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAACklEQV'
    'QImWNoAAAAggCByxOyYQAAAABJRU5ErkJggg=='
)

# Адреса страниц
URL_RECIPES = '/api/recipes/'
URL_GET_RECIPE = '/api/recipes/{id}/'
URL_GET_FRONT_RECIPE = '/recipes/{id}/'
URL_GET_SHORT_LINK = '/api/recipes/{id}/get-link/'
URL_SHORT_LINK = '/s/{uuid}/'

# Структуры для перебора
_INGREDIENTS = [
    {'id': 1, 'amount': 10},
    {'id': 2, 'amount': 20}
]
_TAGS = [1, 2]
_NAME = 'Пример невалидного рецепта'
_TEXT = 'Чисто для тестов.'
_COOKING_TIME = 90

BODY_UPDATE_VALID = {
    'ingredients': [{'id': 1, 'amount': 15}],
    'tags': [1, 2, 3],
    'image': IMAGE,
    'name': 'Обновленный рецепт',
    'text': 'Проверка обновления.',
    'cooking_time': 15
}

BODY_POST_AND_PATH_BAD_REQUESTS = {
    'without_ingredients': {
        'tags': _TAGS,
        'image': IMAGE,
        'name': _NAME,
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    },
    'without_tags': {
        'ingredients': _INGREDIENTS,
        'image': IMAGE,
        'name': _NAME,
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    },
    'empty_ingredients': {
        'ingredients': [],
        'tags': _TAGS,
        'image': IMAGE,
        'name': _NAME,
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    },
    'empty_tags': {
        'ingredients': _INGREDIENTS,
        'tags': [],
        'image': IMAGE,
        'name': _NAME,
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    },
    'empty_image': {
        'ingredients': _INGREDIENTS,
        'tags': _TAGS,
        'image': '',
        'name': _NAME,
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    },
    'empty_name': {
        'ingredients': _INGREDIENTS,
        'tags': _TAGS,
        'image': IMAGE,
        'name': '',
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    },
    'empty_text': {
        'ingredients': _INGREDIENTS,
        'tags': _TAGS,
        'image': IMAGE,
        'name': _NAME,
        'text': '',
        'cooking_time': _COOKING_TIME
    },
    'empty_string_as_cooking_time': {
        'ingredients': _INGREDIENTS,
        'tags': _TAGS,
        'image': IMAGE,
        'name': _NAME,
        'text': _TEXT,
        'cooking_time': ''
    },
    'non_existing_ingredients': {
        'ingredients': [{'id': 9876, 'amount': 25}],
        'tags': _TAGS,
        'image': IMAGE,
        'name': _NAME,
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    },
    'non_existing_tags': {
        'ingredients': _INGREDIENTS,
        'tags': [9876],
        'image': IMAGE,
        'name': _NAME,
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    },
    'repetitive_ingredients': {
        'ingredients': [
            {'id': 1, 'amount': 10},
            {'id': 1, 'amount': 20},
        ],
        'tags': _TAGS,
        'image': IMAGE,
        'name': _NAME,
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    },
    'repetitive_tags': {
        'ingredients': _INGREDIENTS,
        'tags': [1, 1],
        'image': IMAGE,
        'name': _NAME,
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    },
    'ingredients_amount_less_than_one': {
        'ingredients': [{'id': 1, 'amount': 0}],
        'tags': _TAGS,
        'image': IMAGE,
        'name': _NAME,
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    },
    'cooking_time_less_than_one': {
        'ingredients': _INGREDIENTS,
        'tags': _TAGS,
        'image': IMAGE,
        'name': _NAME,
        'text': _TEXT,
        'cooking_time': 0
    },
    'too_long_name': {
        'ingredients': _INGREDIENTS,
        'tags': _TAGS,
        'image': IMAGE,
        'name': (
            'Старинный рецепт, передаваемый из поколения в поколение через '
            'сказки и народные предания. Немногие могут правильно его '
            'приготовить. Большая сложность обусловлена названием, которое '
            'длиннее 256 символов, что существенно усложняет его '
            'запоминание. Дерзайте!!!'
        ),
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    }
}
BODY_ONLY_POST_BAD_REQUEST = {
    'without_image': {
        'ingredients': _INGREDIENTS,
        'tags': _TAGS,
        'name': _NAME,
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    },
    'without_name': {
        'ingredients': _INGREDIENTS,
        'tags': _TAGS,
        'image': IMAGE,
        'text': _TEXT,
        'cooking_time': _COOKING_TIME
    },
    'without_text_field': {
        'ingredients': _INGREDIENTS,
        'tags': _TAGS,
        'image': IMAGE,
        'name': _NAME,
        'cooking_time': _COOKING_TIME
    },
    'without_cooking_time': {
        'ingredients': _INGREDIENTS,
        'tags': _TAGS,
        'image': IMAGE,
        'name': _NAME,
        'text': _TEXT,
    }
}

# Схемы валидации данных в ответах методов
RESPONSE_SCHEMA_SHORT_RECIPE = {
    "type": "object",
    "properties": {
        "id": {"type": "number"},
        "name": {"type": "string"},
        "image": {"type": "string"},
        "cooking_time": {"type": "number"}
    },
    "required": ["id", "name", "image", "cooking_time"],
    "additionalProperties": False
}

RESPONSE_SCHEMA_SHORT_LINK = {
    "type": "object",
    "properties": {
        "short-link": {"type": "string"}
    },
    "required": ["short-link"],
    "additionalProperties": False,
}

RESPONSE_SCHEMA_RECIPE = {
    "type": "object",
    "properties": {
        "id": {"type": "number"},
        "tags": {
            "type": "array",
            "items": RESPONSE_SCHEMA_TAG
        },
        "author": RESPONSE_SCHEMA_USER,
        "ingredients": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "number"},
                    "name": {"type": "string"},
                    "measurement_unit": {"type": "string"},
                    "amount": {"type": "number"}
                },
                "required": ["id", "name", "measurement_unit", "amount"],
                "additionalProperties": False
            }
        },
        "is_favorited": {"type": "boolean"},
        "is_in_shopping_cart": {"type": "boolean"},
        "name": {"type": "string"},
        "image": {"type": "string"},
        "text": {"type": "string"},
        "cooking_time": {"type": "number"}
    },
    "required": [
        "id", "tags", "author", "ingredients", "is_favorited",
        "is_in_shopping_cart", "name", "image", "text", "cooking_time"
    ],
    "additionalProperties": False
}

RESPONSE_SCHEMA_RECIPES = {
    "type": "object",
    "required": ["count", "next", "previous", "results"],
    "additionalProperties": False,
    "properties": {
        "count": {"type": "number"},
        "next": {"type": ["string", "null"]},
        "previous": {"type": ["string", "null"]},
        "results": {
            "type": "array",
            "items": RESPONSE_SCHEMA_RECIPE
        }
    }
}
