# 🐳 Docker Compose для RAG Web UI

Этот набор файлов позволяет запустить веб-интерфейс RAG системы в Docker контейнерах с поддержкой как CPU, так и GPU.

## 📋 Требования

### Для CPU версии:
- Docker
- Docker Compose
- Минимум 4GB RAM

### Для GPU версии:
- Docker с поддержкой NVIDIA Container Toolkit
- NVIDIA GPU с поддержкой CUDA 12.1+
- nvidia-docker2 или Docker с GPU поддержкой

## 🚀 Быстрый старт

### CPU версия (рекомендуется для тестирования)

```bash
# Запуск CPU версии
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

Веб-интерфейс будет доступен по адресу: http://localhost:5000

### GPU версия (для продакшена)

```bash
# Запуск GPU версии
docker-compose -f docker-compose.gpu.yml up -d

# Просмотр логов
docker-compose -f docker-compose.gpu.yml logs -f

# Остановка
docker-compose -f docker-compose.gpu.yml down
```

## ⚙️ Конфигурация

### Переменные окружения

- `TEST_PROJ_PATH` - путь к проекту для анализа (по умолчанию: `/app/project`)
- `PYTHONPATH` - путь к Python модулям (по умолчанию: `/app`)
- `FLASK_ENV` - режим Flask (по умолчанию: `production`)
- `CUDA_VISIBLE_DEVICES` - GPU устройства для GPU версии (по умолчанию: `0`)

### Volumes

- `.:/app/project:ro` - монтирует текущую директорию как проект для анализа
- `./models:/app/models:ro` - монтирует папку с моделями (если есть)

## 🔧 Настройка

### 1. Подготовка проекта

Убедитесь, что в текущей директории есть файлы RAG системы:
- `web_ui.py`
- `rag_*.py` файлы
- `requirements.txt`

### 2. Модели (опционально)

Если у вас есть предобученные модели, поместите их в папку `models/`:

```bash
mkdir models
# Скопируйте ваши модели в models/
```

### 3. Переменные окружения

Скопируйте файл `docker.env` и настройте под свои нужды:

```bash
# Скопируйте файл конфигурации
cp docker.env .env

# Отредактируйте переменные если нужно
# nano .env
```

Или используйте готовый файл `docker.env` как есть.

## 📊 Мониторинг

### Проверка состояния

```bash
# Статус контейнеров
docker-compose ps

# Логи приложения
docker-compose logs rag-web-cpu

# Использование ресурсов
docker stats
```

### Health Check

Контейнеры автоматически проверяют свое состояние через HTTP endpoint `/health`.

## 🐛 Устранение неполадок

### CPU версия

1. **Ошибка памяти**: Увеличьте лимиты Docker или используйте GPU версию
2. **Медленная работа**: Это нормально для CPU версии, рассмотрите GPU версию

### GPU версия

1. **CUDA не найдена**: Убедитесь что установлен NVIDIA Container Toolkit
2. **GPU недоступна**: Проверьте `nvidia-smi` и права доступа к GPU
3. **Ошибка сборки**: Убедитесь что CUDA версия совместима

### Общие проблемы

1. **Порт занят**: Измените порт в docker-compose.yml
2. **Модели не загружаются**: Проверьте монтирование volumes
3. **Медленный старт**: Первый запуск может занять время для загрузки моделей

## 🔄 Обновление

```bash
# Остановка
docker-compose down

# Пересборка образов
docker-compose build --no-cache

# Запуск
docker-compose up -d
```

## 📝 Логи

```bash
# Все логи
docker-compose logs

# Логи конкретного сервиса
docker-compose logs rag-web-cpu

# Следить за логами в реальном времени
docker-compose logs -f rag-web-cpu
```

## 🧹 Очистка

```bash
# Остановка и удаление контейнеров
docker-compose down

# Удаление образов
docker-compose down --rmi all

# Полная очистка (включая volumes)
docker-compose down -v --rmi all
```

## 📈 Производительность

### CPU версия
- ✅ Быстрый старт
- ✅ Низкое потребление ресурсов
- ❌ Медленная обработка
- ❌ Ограниченная память

### GPU версия
- ✅ Быстрая обработка
- ✅ Больше памяти
- ❌ Требует GPU
- ❌ Больше потребление ресурсов

## 🆘 Поддержка

При возникновении проблем:

1. Проверьте логи: `docker-compose logs`
2. Убедитесь что все зависимости установлены
3. Проверьте доступность GPU (для GPU версии)
4. Убедитесь что порты не заняты

## 📚 Дополнительно

- [Docker Compose документация](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [PyTorch CUDA поддержка](https://pytorch.org/get-started/locally/)
