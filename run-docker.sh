#!/bin/bash

# RAG Web UI Docker Runner
# Удобный скрипт для запуска RAG системы в Docker

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
print_message() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Функция проверки Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker не установлен!"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose не установлен!"
        exit 1
    fi
}

# Функция проверки GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            print_success "NVIDIA GPU обнаружена!"
            return 0
        fi
    fi
    print_warning "NVIDIA GPU не обнаружена или недоступна"
    return 1
}

# Функция показа помощи
show_help() {
    echo "🐳 RAG Web UI Docker Runner"
    echo ""
    echo "Использование: $0 [ОПЦИЯ]"
    echo ""
    echo "Опции:"
    echo "  cpu     Запустить CPU версию (по умолчанию)"
    echo "  gpu     Запустить GPU версию"
    echo "  stop    Остановить все контейнеры"
    echo "  logs    Показать логи"
    echo "  status  Показать статус контейнеров"
    echo "  clean   Очистить все контейнеры и образы"
    echo "  help    Показать эту справку"
    echo ""
    echo "Примеры:"
    echo "  $0 cpu          # Запустить CPU версию"
    echo "  $0 gpu          # Запустить GPU версию"
    echo "  $0 logs         # Показать логи"
    echo "  $0 stop         # Остановить"
}

# Функция запуска CPU версии
start_cpu() {
    print_message "Запуск CPU версии..."
    docker-compose up -d
    print_success "CPU версия запущена!"
    print_message "Веб-интерфейс доступен по адресу: http://localhost:5000"
}

# Функция запуска GPU версии
start_gpu() {
    if check_gpu; then
        print_message "Запуск GPU версии..."
        docker-compose -f docker-compose.gpu.yml up -d
        print_success "GPU версия запущена!"
        print_message "Веб-интерфейс доступен по адресу: http://localhost:5000"
    else
        print_error "GPU недоступна! Используйте CPU версию: $0 cpu"
        exit 1
    fi
}

# Функция остановки
stop_containers() {
    print_message "Остановка контейнеров..."
    docker-compose down 2>/dev/null || true
    docker-compose -f docker-compose.gpu.yml down 2>/dev/null || true
    print_success "Контейнеры остановлены!"
}

# Функция показа логов
show_logs() {
    print_message "Показ логов..."
    if docker-compose ps | grep -q "Up"; then
        docker-compose logs -f
    elif docker-compose -f docker-compose.gpu.yml ps | grep -q "Up"; then
        docker-compose -f docker-compose.gpu.yml logs -f
    else
        print_warning "Нет запущенных контейнеров"
    fi
}

# Функция показа статуса
show_status() {
    print_message "Статус контейнеров:"
    echo ""
    docker-compose ps
    echo ""
    docker-compose -f docker-compose.gpu.yml ps
}

# Функция очистки
clean_all() {
    print_warning "Очистка всех контейнеров и образов..."
    docker-compose down --rmi all -v 2>/dev/null || true
    docker-compose -f docker-compose.gpu.yml down --rmi all -v 2>/dev/null || true
    print_success "Очистка завершена!"
}

# Основная логика
main() {
    check_docker
    
    case "${1:-cpu}" in
        "cpu")
            start_cpu
            ;;
        "gpu")
            start_gpu
            ;;
        "stop")
            stop_containers
            ;;
        "logs")
            show_logs
            ;;
        "status")
            show_status
            ;;
        "clean")
            clean_all
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Неизвестная опция: $1"
            show_help
            exit 1
            ;;
    esac
}

# Запуск
main "$@"
