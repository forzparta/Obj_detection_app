import sys
from typing import List

import numpy as np
import torch
from app.helpers.utils import letterbox, non_max_suppression, scale_boxes
from app.configs.model_config import load_config, YOLOScores, BoundingBox


class Yolov8_client:
    def __init__(
        self,
        config_path="app/configs",
        url="triton:",
        model_name="yolov8n",
        conf_thresh=0.5,
        iou_thresh = 0.45,
        grpc=True
    ) -> None:
        super(Yolov8_client).__init__()
        self.model_name = model_name

        config = load_config(f"{config_path}/{model_name}.yaml")
        
        self.input_width = config.input_width
        self.input_height = config.input_height
        self.batch_size = config.batch_size
        self.input_shape = [self.batch_size, 3, self.input_height, self.input_width]
        self.input_name = config.input_name
        self.output_name = config.output_name
        self.output_size = config.output_size
        self.np_dtype = np.float32
        self.fp = config.np_dtype

        self.conf_thresh = conf_thresh
        self.triton_client = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.iou_thresh = iou_thresh

        if(grpc):
            import tritonclient.grpc as trclient
            port = '8001'
        else:
            import tritonclient.http as trclient
            port = '8000'
            
        self.protocolTriton = trclient

        self.init_triton_client(url+port)
        self.test_predict()

    def init_triton_client(self, url: str) -> None:
        try:
            triton_client = self.protocolTriton.InferenceServerClient(
                url=f"{url}",
                verbose=False,
                ssl=False,
            )

        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit()
        self.triton_client = triton_client


    def test_predict(self) -> None:
        input_images = np.zeros((*self.input_shape,), dtype=self.np_dtype)
        _ = self._predict(input_images)


    def _predict(self, input_images: np.ndarray) -> torch.tensor:
        inputs = [self.protocolTriton.InferInput(self.input_name, [*input_images.shape], self.fp)]
        inputs[0].set_data_from_numpy(input_images)
        outputs = [self.protocolTriton.InferRequestedOutput(self.output_name)]

        results = self.triton_client.infer(
            model_name=self.model_name, inputs=inputs, outputs=outputs
        )
        return torch.from_numpy(np.copy(results.as_numpy(self.output_name))).to(self.device)

    
    def postprocess(self, preds: torch.Tensor, origin_h: int, origin_w: int) -> YOLOScores:
        preds = non_max_suppression(preds, self.conf_thresh, self.iou_thresh)

        for pred in preds:
            boxes = []
            if len(pred):
                pred[:, :4] = scale_boxes(
                    (self.input_height, self.input_width), pred[:, :4], (origin_h, origin_w)
                )
                for box in pred[:, :4].cpu().numpy():
                    boxes.append(BoundingBox(xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3]))

            scores = pred[:, 4].cpu().numpy().tolist() if len(pred) else []
            class_ids = pred[:, 5].cpu().numpy().tolist() if len(pred) else []
            
            detection = YOLOScores(boxes=boxes, scores=scores, class_ids=class_ids)
        
        return detection


    def preprocess(self, img: np.ndarray, stride: int = 32) -> np.ndarray:
        img = letterbox(img, max(self.input_width, self.input_height), stride=stride, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img /= 255.0  # Normalize to 0.0 - 1.0
        img = img.reshape([1, *img.shape])  # Add batch dimension
        return img


    def predict(self, image: np.ndarray) -> List[YOLOScores]:
        processed_image = self.preprocess(image)
        pred = self._predict(processed_image)
        return self.postprocess(pred, image.shape[0], image.shape[1])
