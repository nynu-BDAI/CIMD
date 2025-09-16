import argparse
import base64
import time
import random
import numpy as np
import os
import re
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='/mnt/Data/THINGS-EEG2/', type=str)
args = parser.parse_args()
print('Generate text descriptions using DeepSeek-VL <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

class ImageDescriptionGenerator:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.requests_per_minute = 500
        self.last_request_time = 0
        self.min_request_interval = 60.0 / self.requests_per_minute

    def image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_string}"

    def get_image_description(self, image_path, text_query):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
        time.sleep(random.uniform(0.1, 0.5))
        base64_image = self.image_to_base64(image_path)
        try:
            response = self.client.chat.completions.create(
                model="deepseek-vl/deepseek-vl-chat",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image}
                            },
                            {
                                "type": "text",
                                "text": text_query
                            }
                        ]
                    }
                ],
            )
            result = response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred while calling the API: {e}")
            result = "Error generating description."

        # 更新最后一次请求的时间
        self.last_request_time = time.time()
        return result


# --- API ---
api_key = ""
base_url = ""
image_desc_generator = ImageDescriptionGenerator(api_key, base_url)
img_set_dir = os.path.join(args.project_dir, 'Image_set/image')
img_partitions = os.listdir(img_set_dir)

for p in img_partitions:
    part_dir = os.path.join(img_set_dir, p)
    image_list = []
    for root, dirs, files in os.walk(part_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                image_list.append(os.path.join(root, file))
    image_list.sort()
    save_text_dir = os.path.join(args.project_dir, 'Description', 'deepseek_vl_global', p)
    if not os.path.isdir(save_text_dir):
        os.makedirs(save_text_dir)
        print(f"Created directory: {save_text_dir}")

    print(f"\nProcessing partition: {p} ({len(image_list)} images)")

    for i, image_path in enumerate(image_list):
        label_text = os.path.basename(os.path.dirname(image_path))[6:]
        prompt = f"This is a picture of {label_text}. what does this picture describe overall in one or two sentence?"    # global
        # prompt = f"This is a picture of {label_text}. What does the color in this picture include in one sentence?"     # color
        # prompt = f"This is a picture of {label_text}. What emotions does this picture evoke in people in one sentence?" # emotion

        print(f"  [{i + 1}/{len(image_list)}] Generating description for: {os.path.basename(image_path)}")
        response_text = image_desc_generator.get_image_description(image_path, prompt)

        response_text = re.sub('[\u4e00-\u9fa5]', '', response_text)
        response_text = re.sub(r'\d+', '', response_text)
        response_text = re.sub(r"[^A-Za-z,.!?\' ]", "", response_text)
        response_text = re.sub(r'\s+', ' ', response_text).strip()
        response_text = response_text[:350]

        file_name = p + '_' + format(i + 1, '07') + '.txt'
        save_path = os.path.join(save_text_dir, file_name)
        np.savetxt(save_path, [response_text], fmt='%s', encoding='utf-8')

print("\nProcessing complete.")