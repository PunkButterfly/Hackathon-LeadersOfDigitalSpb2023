FROM python:3.8.9

WORKDIR container/

COPY . .

#COPY ./requirements.txt /code/requirements.txt

RUN pip install -r requirements.txt

CMD python app/main.py
