import os
from celery import Celery
from dotenv import dotenv_values

config = {
    **dotenv_values(".env"),  # load general environment variables
    **os.environ,  # override loaded values with environment variables
}


app = Celery(
    "celery_task_app",
    #  broker='amqp://', # default for localhost
    #  broker='amqp://guest:guest@rabbitmq:5672', # default for docker
    broker=config["CELERY_BROKER"],
    backend=config["CELERY_BACKEND"],
    include=["app.celery_task_app.tasks"],
)


if __name__ == '__main__':
    app.start()

