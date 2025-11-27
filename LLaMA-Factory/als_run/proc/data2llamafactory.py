import json
import random
import os
import datasets
from tqdm import tqdm
from pathlib import Path


if __name__ == "__main__":
    data_name = "ShaoRun/RS-EoT-4K"
    data = datasets.load_dataset(data_name)['train']

    save_root = "/path/to/save"

    Path(save_root).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(save_root, "images")).mkdir(exist_ok=True, parents=True)
    
    res = []
    for i, item in tqdm(enumerate(data)):
        query = item['query']
        answer = item['response']
        img_path = os.path.join(save_root, 'images', f'{i}.jpg')
        if not os.path.exists(img_path):
            item['image'].save(img_path)
        res.append({
            "messages": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer}
            ],
            "images": [img_path]
        })

    print(random.choice(res))
    
    with open(os.path.join(save_root, 'RS-EoT-4K-llamafactory.json'), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)