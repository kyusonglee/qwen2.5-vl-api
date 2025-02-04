import requests
import json

# Base URL for the API
base_url = "http://localhost:7860"

# Test the GET /predict endpoint
def test_get_predict():
    url = f"{base_url}/predict"
    params = {
        "image_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        "prompt": "describe"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        print("GET /predict Response:", response.json())
    else:
        print("GET /predict Error:", response.status_code, response.text)

# Test the POST /predict endpoint
def test_post_predict():
    url = f"{base_url}/predict"
    payload = {
        "image_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        "prompt": "describe"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        print("POST /predict Response:", response.json())
    else:
        print("POST /predict Error:", response.status_code, response.text)

# Test the POST /chat endpoint
def test_post_chat():
    url = f"{base_url}/chat"
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant with vision abilities."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                    {"type": "text", "text": "describe"}
                ]
            }
        ]
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        print("POST /chat Response:", response.json())
    else:
        print("POST /chat Error:", response.status_code, response.text)

if __name__ == "__main__":
    test_get_predict()
    test_post_predict()
    test_post_chat()
