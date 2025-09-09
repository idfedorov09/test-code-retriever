# Для доступа к админке можно использовать тестовые учетные данные:
# Email: admin@admin.ru
# Пароль: admin

# ip http://51.250.99.113:8080/

# Foodgram - Продуктовый помощник

![Python](https://img.shields.io/badge/python-3.9-blue?logo=python)
![Django](https://img.shields.io/badge/django-3.2-success?logo=django)
![Django REST Framework](https://img.shields.io/badge/drf-3.12-orange?logo=django)
![Docker](https://img.shields.io/badge/docker-20.10-blue?logo=docker)
![PostgreSQL](https://img.shields.io/badge/postgresql-13-blue?logo=postgresql)

## Описание проекта

Foodgram - это веб-приложение, созданное в рамках учебного курса Яндекс Практикума. Платформа позволяет пользователям публиковать рецепты, добавлять их в избранное, подписываться на других авторов и создавать список покупок на основе выбранных блюд.

Основные возможности:
- Публикация рецептов с изображениями
- Фильтрация рецептов по тегам
- Добавление рецептов в избранное
- Подписка на авторов
- Создание и скачивание списка покупок
- Поиск по ингредиентам при добавлении в рецепт

## Технологии

- Python 3.9
- Django 3.2
- Django REST Framework 3.12
- PostgreSQL 13
- Docker & Docker Compose
- Nginx
- Gunicorn
- React (для фронтенда)

## Установка и запуск

### Локальное развертывание с Docker

Склонируйте репозиторий:
```bash
git clone https://github.com/arefiture/foodgram.git
cd foodgram

Создайте файл .env в корне проекта:

# Создайте файл .env в корне проекта:
# .env
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
USE_PGSQL=True
POSTGRES_DB=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
DB_HOST=db
DB_PORT=5432

# Запустите проект с помощью Docker Compose:
docker compose -f docker-compose.production.yml up -d

# Примените миграции:
docker compose -f docker-compose.production.yml exec backend python manage.py migrate

# Соберите статику:
docker compose -f docker-compose.production.yml exec backend python manage.py collectstatic

# Создайте суперпользователя (опционально):
docker compose -f docker-compose.production.yml exec backend python manage.py createsuperuser

# Доступ к приложению
# После запуска приложение будет доступно по адресу:
# Фронтенд: http://localhost:8080 (или ваш домен)
# Админка Django: http://localhost:8080/admin/

# Документация API доступна по адресу: http://localhost:8080/api/docs/ или http://localhost:8080/api/docs/redoc.html

# Остановка проекта
docker compose -f docker-compose.production.yml down
