# RAG Web UI Docker Makefile
# Удобные команды для работы с Docker

.PHONY: up

upd_and_up:
	git fetch && git pull --rebase && docker compose -f docker-compose.gpu.yml up -d --build rag-web-gpu