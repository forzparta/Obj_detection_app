import asyncio
import json

from app.celery_app import app, config
from app.configs.model_config import DetectionPayload, YOLOScores
from app.helpers.utils import string2image
from app.models_clients.yolov8 import Yolov8_client


@app.task()
def predict_image(payload: str) -> YOLOScores:
    async def async_predict_image():
        data_dict = json.loads(payload)
        data = DetectionPayload(**data_dict)
        image = string2image(data.image)
        # change how port is handled
        yolov8_client = Yolov8_client(
            url=config["TRITON"],
            model_name=data.model_name,
            conf_thresh=data.conf_threshold,
            iou_thresh=data.iou_threshold,
            grpc=True,
        )
        result: YOLOScores = yolov8_client.predict(image)
        return result.model_dump_json()

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_predict_image())
