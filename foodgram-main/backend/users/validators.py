from typing import Optional

from rest_framework.exceptions import ValidationError


class SubscribeUniqueValidator:
    """
    Валидатор для сравнения подписчика и автора.

    В случае если они равны вызывает исключение, т.к.
    подпись на самого себя не имеет смысла.
    """

    message = 'Невозможно подписаться на самого себя'

    def __init__(self, fields: list, message: Optional[str] = None):
        self.fields = fields
        self.message = message or self.message

    def __call__(self, attrs: dict):
        user = attrs.get('user')
        author_recipe = attrs.get('author_recipe')

        if user == author_recipe:
            raise ValidationError(self.message)
