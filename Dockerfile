FROM python:3.10-slim

RUN pip install poetry==1.7.1
WORKDIR /code
COPY poetry.lock pyproject.toml /code/
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi
COPY app /code/app
COPY .env /code/.env
COPY start.sh /code/start.sh
RUN chmod +x /code/start.sh
CMD ["/code/start.sh"]