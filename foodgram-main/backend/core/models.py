from django.db import models

from core.utils import to_snake_case


class PrefixedDBModel(models.Model):
    """Заготовка для установки префикса пред названием таблиц."""

    class Meta:
        abstract = True

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        table_name = to_snake_case(cls.__name__)
        prefix_name = cls.prefix_name
        cls.Meta.db_table = f'{prefix_name}_{table_name}'
