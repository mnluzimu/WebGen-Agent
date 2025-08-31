import requests
import json
from PIL import Image
import base64
 
 
# 1.url
url = '/mnt/cache/luzimu/code_agent/APP-Bench-Remote/src/deploy/test_api.py/mnt/cache/luzimu/code_agent/APP-Bench-Remote/src/deploy/test_api.py'
 
 
# 2.data
data = {"model": "/mnt/cache/luzimu/code_agent/outs/Aguvis-7B-720P",
        "messages": [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                     {"role": "user",
                      "content": [
                          {"type": "image_url", "image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
                          {"type": "text", "text": "Describe this image."},],}],
        "temperature": 0.7,"top_p": 0.8,"repetition_penalty": 1.05,"max_tokens": 512}
 
 
# 3.将字典转换为 JSON 字符串
json_payload = json.dumps(data)
 
 
# 4.发送 POST 请求
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json_payload, headers=headers)

print(response)
 
# 5.打印响应内容
print(response.json().get("choices", [])[0].get("message", []).get("content", []))      # 命令行启动，用这个打印
# print(response.json())