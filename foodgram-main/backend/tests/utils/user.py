# Константы из postman
USERNAME = 'vasya.ivanov'
PASSWORD = 'MySecretPas$word'
EMAIL = 'vivanov@yandex.ru'
AVATAR = (
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAgMAAABieywaAAA'
    'ACVBMVEUAAAD///9fX1/S0ecCAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAACklEQVQImWNo'
    'AAAAggCByxOyYQAAAABJRU5ErkJggg=='
)
TOO_LONG_EMAIL = (
    'i_have_never_seen_an_email_address_longer_than_two_hundred_and_fifty_'
    'four_characters_and_it_was_difficult_to_come_up_with_it_so_in_the_'
    'second_part_just_the_names_of_some_mail_services@yandex-google-yahoo-'
    'mailgun-protonmail-mailru-outlook-icloud-aol-neo.ru'
)
TOO_LONG_USERNAME = (
    'the-username-that-is-150-characters-long-and-should-not-pass-validation-'
    'if-the-serializer-is-configured-correctly-otherwise-the-current-test-'
    'will-fail-'
)
NEW_PASSWORD = 'thi$Pa$$w0rdW@sCh@nged'
SECOND_USER_USERNAME = 'second-user'
SECOND_USER_EMAIL = 'second_user@email.org'
THIRD_USER_USERNAME = 'third-user'
THIRD_USER_EMAIL = 'third-user@user.ru'

# Адреса страниц
URL_AVATAR = '/api/users/me/avatar/'
URL_CREATE_USER = '/api/users/'
URL_GET_USER = URL_CREATE_USER + '{id}/'
URL_LOGIN = '/api/auth/token/login/'
URL_LOGOUT = '/api/auth/token/logout/'
URL_ME = '/api/users/me/'
URL_SET_PASSWORD = '/api/users/set_password/'

# Структуры для перебора
INVALID_USER_DATA_FOR_LOGIN = [
    {
        'email': SECOND_USER_USERNAME,
        'password': 'randomPassword'
    },
    {
        'password': PASSWORD
    },
    {
        'email': SECOND_USER_USERNAME
    }
]

INVALID_USER_DATA_FOR_REGISTER = [
    {
        'username': 'NoEmail',
        'first_name': 'No',
        'last_name': 'Email',
        'password': PASSWORD
    },
    {
        'email': 'no-username@user.ru',
        'first_name': 'Username',
        'last_name': 'NotProvided',
        'password': PASSWORD
    },
    {
        'username': 'NoFirstName',
        'email': 'no-first-name@user.ru',
        'last_name': 'NoFirstName',
        'password': PASSWORD
    },
    {
        'username': 'NoLastName',
        'email': 'no-last-name@user.ru',
        'first_name': 'NoLastName',
        'password': PASSWORD
    },
    {
        'username': 'NoPassword',
        'email': 'no-pasword@user.ru',
        'first_name': 'NoPassword',
        'last_name': 'NoPassword'
    },
    {
        'username': 'TooLongEmail',
        'email': TOO_LONG_EMAIL,
        'first_name': 'TooLongEmail',
        'last_name': 'TooLongEmail',
        'password': PASSWORD
    },
    {
        'username': TOO_LONG_USERNAME,
        'email': 'too-long-username@user.ru',
        'first_name': 'TooLongUsername',
        'last_name': 'TooLongUsername',
        'password': PASSWORD
    },
    {
        'username': 'TooLongFirstName',
        'email': 'too-long-firt-name@user.ru',
        'first_name': TOO_LONG_USERNAME,
        'last_name': 'TooLongFirstName',
        'password': PASSWORD
    },
    {
        'username': 'TooLongLastName',
        'email': 'too-long-last-name@user.ru',
        'first_name': 'TooLongLastName',
        'last_name': TOO_LONG_USERNAME,
        'password': PASSWORD
    },
    {
        'username': 'InvalidU$ername',
        'email': 'invalid-username@user.ru',
        'first_name': 'Invalid',
        'last_name': 'Username',
        'password': PASSWORD
    }
]

IN_USE_USER_DATA_FOR_REGISTER = [
    {
        'email': EMAIL,
        'username': 'EmailInUse',
        'first_name': 'Email',
        'last_name': 'InUse',
        'password': PASSWORD
    },
    {
        'email': 'username-in-use@user.ru',
        'username': USERNAME,
        'first_name': 'Username',
        'last_name': 'InUse',
        'password': PASSWORD
    }
]

FIRST_VALID_USER = {
    'email': EMAIL,
    'username': USERNAME,
    'first_name': 'Вася',
    'last_name': 'Иванов',
    'password': PASSWORD
}

SECOND_VALID_USER = {
    'email': SECOND_USER_EMAIL,
    'username': SECOND_USER_USERNAME,
    'first_name': 'Андрей',
    'last_name': 'Макаревский',
    'password': PASSWORD,
    'avatar': AVATAR
}

THIRD_VALID_USER = {
    'email': THIRD_USER_EMAIL,
    'username': THIRD_USER_USERNAME,
    'first_name': 'Гордон',
    'last_name': 'Рамзиков',
    'password': PASSWORD
}

# Схемы валидации данных в ответах методов
RESPONSE_SCHEMA_TOKEN = {
    "type": "object",
    "properties": {
        "auth_token": {"type": "string"},
    },
    "required": ["auth_token"],
    "additionalProperties": False
}

RESPONSE_SCHEMA_AVATAR = {
    "type": "object",
    "properties": {
        "avatar": {"type": "string"}
    },
    "required": ["avatar"],
    "additionalProperties": False,
}

RESPONSE_SCHEMA_USER = {
    "type": "object",
    "properties": {
        "id": {"type": "number"},
        "username": {"type": "string"},
        "first_name": {"type": "string"},
        "last_name": {"type": "string"},
        "email": {"type": "string"},
        "is_subscribed": {"type": "boolean"},
        "avatar": {"type": ["string", "null"]}
    },
    "required": [
        "id", "username", "first_name", "last_name", "email",
        "is_subscribed", "avatar"
    ],
    "additionalProperties": False
}

RESPONSE_SCHEMA_USERS = {
    "type": "object",
    "properties": {
        "count": {"type": "number"},
        "next": {"type": ["string", "null"]},
        "previous": {"type": ["string", "null"]},
        "results": {
            "type": "array",
            "items": {
                "type": RESPONSE_SCHEMA_USER['type'],
                "properties": RESPONSE_SCHEMA_USER['properties'],
                "required": [
                    "id", "username", "first_name", "last_name",
                    "email", "avatar"
                ],
                "additionalProperties": RESPONSE_SCHEMA_USER[
                    'additionalProperties'
                ]
            }
        }
    },
    "required": ["count", "next", "previous", "results"],
    "additionalProperties": False
}
