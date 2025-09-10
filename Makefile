# RAG Web UI Docker Makefile
# Удобные команды для работы с Docker

.PHONY: help cpu gpu stop logs status clean build-cpu build-gpu

# Показать справку
help:
	@echo "🐳 RAG Web UI Docker Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make cpu        - Запустить CPU версию"
	@echo "  make gpu        - Запустить GPU версию"
	@echo "  make stop       - Остановить все контейнеры"
	@echo "  make logs       - Показать логи"
	@echo "  make status     - Показать статус"
	@echo "  make clean      - Очистить все"
	@echo "  make build-cpu  - Собрать CPU образ"
	@echo "  make build-gpu  - Собрать GPU образ"
	@echo ""

# CPU версия
cpu:
	@echo "🚀 Запуск CPU версии..."
	docker-compose up -d
	@echo "✅ CPU версия запущена! Веб-интерфейс: http://localhost:5000"

# GPU версия
gpu:
	@echo "🚀 Запуск GPU версии..."
	docker-compose -f docker-compose.gpu.yml up -d
	@echo "✅ GPU версия запущена! Веб-интерфейс: http://localhost:5000"

# Остановка
stop:
	@echo "🛑 Остановка контейнеров..."
	docker-compose down 2>/dev/null || true
	docker-compose -f docker-compose.gpu.yml down 2>/dev/null || true
	@echo "✅ Контейнеры остановлены!"

# Логи
logs:
	@echo "📋 Показ логов..."
	@if docker-compose ps | grep -q "Up"; then \
		docker-compose logs -f; \
	elif docker-compose -f docker-compose.gpu.yml ps | grep -q "Up"; then \
		docker-compose -f docker-compose.gpu.yml logs -f; \
	else \
		echo "⚠️  Нет запущенных контейнеров"; \
	fi

# Статус
status:
	@echo "📊 Статус контейнеров:"
	@echo ""
	@docker-compose ps
	@echo ""
	@docker-compose -f docker-compose.gpu.yml ps

# Очистка
clean:
	@echo "🧹 Очистка контейнеров и образов..."
	docker-compose down --rmi all -v 2>/dev/null || true
	docker-compose -f docker-compose.gpu.yml down --rmi all -v 2>/dev/null || true
	@echo "✅ Очистка завершена!"

# Сборка CPU образа
build-cpu:
	@echo "🔨 Сборка CPU образа..."
	docker-compose build --no-cache rag-web-cpu
	@echo "✅ CPU образ собран!"

# Сборка GPU образа
build-gpu:
	@echo "🔨 Сборка GPU образа..."
	docker-compose -f docker-compose.gpu.yml build --no-cache rag-web-gpu
	@echo "✅ GPU образ собран!"

# Пересборка всех образов
rebuild: build-cpu build-gpu
	@echo "✅ Все образы пересобраны!"

# Проверка GPU
check-gpu:
	@echo "🔍 Проверка GPU..."
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		nvidia-smi; \
	else \
		echo "❌ nvidia-smi не найден. GPU недоступна."; \
	fi

# Установка зависимостей
install:
	@echo "📦 Установка зависимостей..."
	@if [ ! -f docker.env ]; then \
		echo "⚠️  Файл docker.env не найден. Создайте его на основе docker.env.example"; \
		exit 1; \
	fi
	@echo "✅ Готово к запуску!"
