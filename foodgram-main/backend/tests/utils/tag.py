from tests.utils.general import CHANGE_METHOD, installation_method_urls

# Адреса страниц
URL_TAGS = '/api/tags/'
URL_GET_TAG = '/api/tags/{id}/'

# Структуры для перебора
TAG_DATA = [
    {'name': 'Десерт', 'slug': 'dessert'},
    {'name': 'Завтрак', 'slug': 'breakfast'},
    {'name': 'Обед', 'slug': 'lunch'},
    {'name': 'Ужин', 'slug': 'dinner'}
]

TAG_SET_SLUGS = {  # SET - не множества, а наборы
    'one_tag': [TAG_DATA[0].get('slug')],
    'two_tags': [
        TAG_DATA[0].get('slug'),
        TAG_DATA[1].get('slug')
    ],
    'three_tags': [
        TAG_DATA[1].get('slug'),
        TAG_DATA[2].get('slug'),
        TAG_DATA[3].get('slug')
    ]
}

DENY_CHANGE_METHOD = installation_method_urls(
    url=URL_TAGS,
    url_detail=URL_GET_TAG.format(id=1),
    dict_method_urls=CHANGE_METHOD
)

# Схемы валидации данных в ответах методов
RESPONSE_SCHEMA_TAG = {
    "type": "object",
    "properties": {
        "id": {"type": "number"},
        "name": {"type": "string"},
        "slug": {"type": "string"},
    },
    "required": ["id", "name", "slug"],
    "additionalProperties": False
}

RESPONSE_SCHEMA_TAGS = {
    "type": "array",
    "items": RESPONSE_SCHEMA_TAG
}
