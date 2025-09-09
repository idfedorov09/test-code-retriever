from tests.utils.recipe import URL_GET_RECIPE, URL_RECIPES

# Адреса страниц
URL_SHOPPING_CART = URL_GET_RECIPE + 'shopping_cart/'
URL_DOWNLOAD_SHOPPING_CART = URL_RECIPES + 'download_shopping_cart/'

# Информация для валидации
ALLOWED_CONTENT_TYPES = (
    'text/plain',
    'application/pdf',
    'text/csv'
)
