FROM python:3.10.6

COPY requirements.txt .
RUN apt-get update && apt-get install libgl1 -y && pip install --no-cache-dir --upgrade pip && pip install -r requirements.txt

RUN mkdir /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app

CMD ["python", "bot.py"]