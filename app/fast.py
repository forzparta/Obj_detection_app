from fastapi import FastAPI

from app.celery_task_app.tasks import predict_image
from app.configs.model_config import DetectionPayload

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/detect")
async def detect(payload: DetectionPayload):
    task = predict_image.delay(payload.model_dump_json())
    result = task.get(timeout=100)

    return result
