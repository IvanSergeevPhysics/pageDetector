FROM python:slim-buster

COPY requirements.txt .
RUN apt-get update && apt-get install libgl1 -y \
&& apt-get install libglib2.0-0 -y \
&& pip install --no-cache-dir --upgrade pip \
&& pip install -r requirements.txt \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app

CMD ["python", "bot.py"]