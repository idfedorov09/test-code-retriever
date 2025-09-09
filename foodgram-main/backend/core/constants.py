# Ограничения для валидаторов
INGREDIENT_AMOUNT_MIN = 1
RECIPE_COOKING_TIME_MIN = 1

# Шаблоны сообщения ошибки
TEMPLATE_MESSAGE_MINIMUM_ONE_ERROR = 'Должен быть хотя бы один {field_name}.'
TEMPLATE_MESSAGE_UNIQUE_ERROR = '{field_name} не должны повторяться.'

# Готовые сообщения ошибки
REPEAT_ADDED_FAVORITE_ERROR = 'Нельзя повторно добавить рецепт в избранные.'
REPEAT_ADDED_SHOPPING_CART_ERROR = 'Нельзя повторно добавить рецепт в корзину.'
MIN_COOKING_TIME_ERROR = (
    f'Время не может быть меньше {RECIPE_COOKING_TIME_MIN} минуты.'
)
MIN_INGREDIENT_AMOUNT_ERROR = (
    f'Количество должно быть равно {INGREDIENT_AMOUNT_MIN} или больше.'
)
USER_EMAIL_ERROR = 'Данный электронный адрес уже используется.'
USER_USERNAME_ERROR = 'Пользователь с таким ником уже существует.'
SUPERUSER_STAFF_ERROR = 'Суперпользователь должен иметь is_staff=True.'

# Названия "схем" - префикс перед названием таблицы
COOKBOOK = 'cookbook'  # Префикс моделей приложения api
AUTH = 'auth'  # Префикс моделей приложения users

# Ограничения по длине текстового поля
LENGTH_CHARFIELD_32 = 32
LENGTH_CHARFIELD_64 = 64
LENGTH_CHARFIELD_128 = 128
LENGTH_CHARFIELD_150 = 150
LENGTH_CHARFIELD_256 = 256

# Адреса
RECIPE_IMAGE_PATH = 'recipes/images/'
RECIPE_DETAIL_URL = '/api/recipes/{pk}/'
FRONTEND_DETAIL_URL = '/recipes/{pk}/'
USER_AVATAR_PATH = 'users/'

# Остальные значения
MAX_LENGTH_SHORT_LINK = 6
