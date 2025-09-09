from typing import Optional

from rest_framework.exceptions import ValidationError


class UniqueDataInManyFieldValidator:
    """Валидатор на уникальность сложных полей."""

    def __init__(
        self, *, field: str, message: str,
        is_dict: bool = False, key: Optional[str] = None
    ):
        self.field = field
        self.message = message
        self.is_dict = is_dict
        if is_dict:
            if not key:
                raise ValueError(
                    {'message': 'Требуется передать поле key для поиска'}
                )
            self.search_field = key

    def __call__(self, value):
        data_list = value.get(self.field)
        data_set = {
            field.get(self.search_field) if self.is_dict else field
            for field in data_list
        }
        if len(data_list) != len(data_set):
            raise ValidationError(self.message)
