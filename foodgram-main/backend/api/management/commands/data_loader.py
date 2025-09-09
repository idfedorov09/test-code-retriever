import csv
import json

from django.core.management.base import BaseCommand
from django.db import models, transaction
from django.utils import termcolors


class Command(BaseCommand):
    data = [
        {
            'file_name': 'ingredients',
            'model': 'api.Ingredient',
            'fields': ['name', 'measurement_unit'],
            'type': 'csv'
        },
        {
            'file_name': 'tags',
            'model': 'api.Tag',
            'fields': ['name', 'slug'],
            'type': 'json'
        }
        # Добавьте другие файлы, если нужно
    ]
    help = "Загрузка данных из CSV и JSON файлов в указанные модели"

    def __init__(self) -> None:
        super().__init__()
        self.style.NOTICE = termcolors.make_style(fg='cyan', opts=('bold',))

    def add_arguments(self, parser):
        parser.add_argument(
            'file_type',
            type=str,
            nargs='?',
            default='all',
            help="Тип файла для загрузки: csv, json, или all (все файлы)"
        )

    @transaction.atomic
    def handle(self, *args, **kwargs):
        from django.apps import apps

        file_type: str = kwargs['file_type'].lower()

        # Определяем, какие типы файлов обрабатывать
        valid_file_types = {'csv', 'json', 'all'}
        if file_type not in valid_file_types:
            self.stderr.write(self.style.ERROR(
                f"Неверный тип: {file_type}. Доступны к выбору: csv, json, "
                "all (по умолчанию, работает как с csv, так и json)."
            ))
            return

        try:
            for entry in self.data:
                if file_type != 'all' and entry['type'] != file_type:
                    self.stderr.write(self.style.WARNING(
                        f"Данные {entry} не подходят для загрузки при "
                        "заданных параметрах."
                    ))
                    continue

                file_path = f"data/{entry['file_name']}.{entry['type']}"
                model = apps.get_model(entry['model'])
                fields: list = entry['fields']
                self.stdout.write(self.style.NOTICE(
                    f"Обработка файла: {file_path} для модели: "
                    f"{entry['model']} ({entry['type']})"
                ))

                # Загружаем данные в зависимости от типа файла
                if entry['type'] == 'csv':
                    self.load_csv(file_path, model, fields)
                elif entry['type'] == 'json':
                    self.load_json(file_path, model, fields)

            self.stdout.write(self.style.SUCCESS(
                f"Данные успешно загружены из: {file_type}")
            )

        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Ошибка загрузки данных: {e}"))

    def load_csv(self, file_path: str, model: models.Model, fields: list):
        """Обработка CSV-файла."""

        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                data = {fields[i]: row[i] for i in range(len(fields))}
                model.objects.update_or_create(
                    **{fields[0]: data[fields[0]]},
                    defaults=data
                )

    def load_json(self, file_path: str, model: models.Model, fields: list):
        """Обработка JSON-файла."""

        with open(file_path, mode='r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                model_data = {
                    field: item[field] for field in fields if field in item
                }
                model.objects.update_or_create(
                    **{fields[0]: model_data[fields[0]]},
                    defaults=model_data
                )
