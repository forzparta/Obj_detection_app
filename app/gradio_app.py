import json
from typing import Dict

import gradio as gr
import requests
import yaml
from helpers.utils import image2string
from PIL import Image, ImageDraw, ImageFont


def draw_boxes_on_image(
    image: Image.Image, results: Dict, config_path: str
) -> Image.Image:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    label_map = config.get("label_map", {})
    colors = config.get("colors", {})

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)

    for box, score, class_id in zip(
        results["boxes"], results["scores"], results["class_ids"]
    ):
        label = label_map.get(str(class_id), "Unknown")
        color = tuple(
            colors.get(label, [255, 0, 0])
        )  # Default to red if no color specified

        # Ensure box coordinates are integers
        xmin, ymin, xmax, ymax = map(
            int, [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
        )

        # Draw the bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)

        text = f"{label}: {score:.2f}"
        text_bbox = draw.textbbox((xmin, ymin - 20), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((xmin, ymin - 20), text, fill="white", font=font)

    return image


def predict(image, conf_threshold, iou_threshold, model_name):
    image_data = image2string(image)
    # Define the payload
    payload = {
        "image": image_data,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "model_name": model_name,
    }

    # TODO change url
    response = requests.post("http://0.0.0.0:3000/detect", json=payload)

    # Debugging prints
    print(f"Response status code: {response.status_code}")
    print(f"Response text: {response.text}")

    # Ensure response is successful
    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"
    text = response.json()
    result_dict = json.loads(text)  # Use json.loads to parse JSON string

    img = draw_boxes_on_image(
        image, result_dict, config_path="app/configs/yolov8n.yaml"
    )
    return [img, result_dict]


iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
        gr.components.Dropdown(
            ["yolov8n", "yolov8m"], value="yolov8n", label="Trained Models"
        ),
    ],
    outputs=[gr.Image(type="pil", label="Result"), gr.JSON(label="Annotation_info")],
    # outputs=gr.JSON(label="Annotation_info"),
    title="Helmet Detection",
    description="Upload images for inference and chose the model to test.",
    examples=[
        [
            "/home/forz/Test_deepL/HelmetDetection/experiment/gradio/samples/005304.jpg",
            0.25,
            0.45,
        ],
        [
            "/home/forz/Test_deepL/HelmetDetection/experiment/gradio/samples/005307.jpg",
            0.25,
            0.45,
        ],
    ],
)

if __name__ == "__main__":
    iface.launch()
