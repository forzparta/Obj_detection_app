import numpy as np
from pydantic import BaseModel
from typing import Dict, List
import yaml


class DetectionPayload(BaseModel):
    image: str
    conf_threshold: float
    iou_threshold: float
    model_name: str


class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    

class YOLOScores(BaseModel):
    boxes: List[BoundingBox]
    scores: List[float]
    class_ids: List[int]


class Config(BaseModel):
    label_map: Dict[str, str]
    colors: Dict[str, List[int]]
    input_width: int
    input_height: int
    batch_size: int
    input_name: str
    output_name: str
    output_size: int
    np_dtype: str


# Function to load configuration from YAML file using Pydantic
def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)