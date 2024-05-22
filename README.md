# Object Detection Application

This repository contains an object detection application using YOLOv8 and Triton Inference Server. The application includes a FastAPI server for handling detection requests, a Celery task queue for asynchronous processing, and a Gradio interface for user interaction.

## Components

1. **FastAPI Server**: Handles HTTP requests for image inference.
2. **Celery Task Queue**: Manages asynchronous processing of image inference tasks.
3. **Triton Inference Server**: Serves the trained models and performs inference.
4. **Gradio Interface**: Provides a web-based UI for uploading images and viewing inference results.
5. **RabbitMQ**: Serves as the message broker for Celery
6. **Docker and Docker Compose**: Containerize and manage the deployment of the application.

## Directory Structure

```bash
Obj_detection_app/
├── app/
│   ├── celery_task_app/
│   │   ├── __init__.py
│   │   └── tasks.py
│   ├── configs/
│   │   ├── __init__.py
│   │   ├── yolov8n.yaml
│   │   └── model_config.py
│   ├── helpers/
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── models_clients/
│   │   ├── __init__.py
│   │   └── yolov8.py
│   ├── celery_app.py
│   ├── fast.py
│   ├── gradio_app.py
├── model_repository/
│   └── yolov8n/
│       ├── 1/model.onnx
│       └── config.pbtxt
├── .env
├── docker-compose.yml
├── Dockerfile
├── poetry.lock
├── pyproject.toml
├── README.md
└── start.sh
```

## Setup and Installation

### Prerequisites

- Docker
- Docker Compose
- Poetry
- NVIDIA GPU with drivers and CUDA installed

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd Obj_detection_app
   ```

2. **Install dependencies:**

   ```bash
   poetry install
   ```

3. **Run the application using Docker Compose:**

   ```bash
   docker-compose up -d
   ```

## Usage

### FastAPI Server

The FastAPI server handles detection requests. You can send a POST request to the `/detect` endpoint with the image and configuration parameters.

### Celery Task Queue

Celery is used to process detection tasks asynchronously. The tasks are defined in `app/celery_task_app/tasks.py`.

### Gradio Interface

Gradio provides a web-based interface for uploading images and viewing detection results. You can start the Gradio interface by running:

```bash
poetry shell
python app/gradio_app.py
```

### Example

1. **Upload an image:**

   - Use the Gradio interface to upload an image.

2. **View results:**
   - The interface will display the image with detected objects and their confidence scores.

## Configuration

### Model Configuration

The model configuration file is located at `app/configs/yolov8n.yaml`. Update this file to change the model parameters and settings.

## Troubleshooting

- **Docker Issues:**
  Ensure Docker and Docker Compose are correctly installed and configured.
- **GPU Issues:**
  Ensure NVIDIA drivers and CUDA are correctly installed.
