cd infra/

if [[ $1 == "start" ]] ; then
    docker compose up -d
elif [[ $1 == "stop" ]] ; then
    docker compose down
elif [[ $1 == "reload" ]] ; then
    docker compose down
    index=1
    while (( index <= ${#} )); do
        if [[ "$1" == "back" ]] ; then
            echo "Перезапуск бека"
            docker image rm infra-backend
        elif [[ "$1" == "front" ]] ; then
            echo "Перезапуск фронта"
            docker image rm infra-frontend
        elif [[ "$1" == "nginx" ]] ; then
            echo "Перезапуск gateway"
            docker image rm nginx:1.25.4-alpine
        fi
        shift
    done
    docker compose up -d
fi