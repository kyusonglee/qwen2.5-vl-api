# Qwen2.5-VL API

This repository implements a vision-enabled chat API using FastAPI, Transformers, and Qwen2.5-VL. The project supports both image and text processing and provides endpoints for generating responses based on visual and textual inputs.

## Features

- **GET /predict**: Accepts URL query parameters (`image_url` and `prompt`) to generate a response.
- **POST /predict**: Accepts a JSON payload (`{"image_url": "...", "prompt": "..."}`) to generate a response.
- **POST /chat**: Accepts a JSON payload with raw `messages`, allowing you full control over the conversation format.

## Setup

### Local Environment

1. **Clone the Repository**
   ```bash
   git clone https://github.com/kyusonglee/qwen2.5-vl-api.git
   cd qwen2.5-vl-api
   ```

2. **Install Dependencies**

   Use Python 3.10 or later and create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Run the API**
   ```bash
   uvicorn main:app --reload
   ```
   The API will run at `http://localhost:8000` by default. Modify the port if necessary.

### Docker

The project includes a `Dockerfile` and a `docker-compose.yaml` for containerized deployment.

1. **Build and Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```
   According to the docker-compose settings, the API is exposed on port `7860`.

### Environment Variables

- **CHECKPOINT:** Path to the model checkpoint. Defaults to `Qwen/Qwen2.5-VL-3B-Instruct` if not set.
- **WORKERS:** Number of Gunicorn workers (when using Docker).

## Usage

### Endpoints

- **GET /predict**

  Example URL:
  ```
  http://localhost:7860/predict?image_url=https://example.com/image.jpg&prompt=describe
  ```

- **POST /predict**

  Example JSON payload:
  ```json
  {
    "image_url": "https://example.com/image.jpg",
    "prompt": "describe"
  }
  ```

- **POST /chat**

  Example JSON payload:
  ```json
  {
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant with vision abilities."
      },
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "https://example.com/image.jpg"},
          {"type": "text", "text": "describe"}
        ]
      }
    ]
  }
  ```

### How It Works

- The model is loaded on startup based on the configured checkpoint.
- Blocking model inference is offloaded to a thread pool executor to keep the FastAPI event loop responsive.
- The API supports both traditional predictions using query parameters as well as more flexible raw message input.

## References

This implementation is based on the guide provided by Hugging Face. You can find the original article [here](https://huggingface.co/blog/ariG23498/qwen25vl-api).

Additionally, more information about Qwen2.5-VL can be found on the [Qwen blog](https://qwenlm.github.io/blog/qwen2.5-vl/).


## License

This project is licensed under the MIT License. 