FROM python:latest

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install -r requirements.txt 

RUN mkdir /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app

CMD ["python", "bot.py"]