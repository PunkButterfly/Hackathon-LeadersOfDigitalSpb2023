# Hackathon-DigitalBreakthroughSaintP
## Сервис

Решение разработано в виде веб-приложения на streamlit, хостится на облачной платформе Yandex.Cloud. Ознакомиться можно по ссылке ССЫЛКА НА СТРИМЛИТ
API реализовано с технологией контейнеризации, поэтому при желании можно развернуть решения на своих вычислительных ресурсах, например с помощью команд ниже.
```
docker pull kuprik01/dh_linux_prod
docker run -p 80:80 kuprik01/dh_linux_prod
```
Помимо веб-версии, также можно напрямую обращаться к API, например: 
```python3
import requests
import json

def parse_result(response):
    status = response['status']
    if not status:
        return "Empty result"
    
    queries = response['query']
    result_list = response['result'] # [sorted_top_for_query_1, sorted_top_for_query_2 ...]

    return status, queries, result_list


url = 'http://158.160.21.26:8503/query'

payload = {"objects": ['екатериненский парк', 'ул думская 18']}
response = requests.post(url, data=json.dumps(payload)).json()

print(parse_result(response))
```
## Навигация
## Как запустить проект?
В проекте исопльзуется Python версии 3.8.9.
Для запуска FastApi нужно усnановить все зависимости из requirements.txt, после чего запустить команду ```uvicorn app.main:app --host 0.0.0.0 --port 8503``` из домашней директории проекта. 

Интерфейс fast-api после этого можно открыть по ссылке http://0.0.0.0:8503/docs.
