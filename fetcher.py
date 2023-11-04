# Tool Name: Fetch Anything From Objaverse
# Version: 0.1
# Author: loping151(Kailing Wong)
# Project Link: https://github.com/
# License: MIT License


import objaverse # tested version: 0.1.7
import objaverse.xl as oxl
import os
import torch
import clip
import logging
import json
import pandas as pd
import shutil
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from renderer import render_stl, render_obj
import random
    
render = {"stl": render_stl, "obj": render_obj}


class Fetcher():
    def __init__(self, config_path) -> None:
        logging.info('reading config file')
        config_data = {}
        try:
            with open(config_path, 'r') as json_file:
                config_data = json.load(json_file)
        except FileNotFoundError:
            logging.warn(f"Config file '{config_path}' not found. Using default config.")
        except json.JSONDecodeError:
            logging.error(f"Error parsing JSON in '{config_path}'. Check the JSON format.")
            exit(1)
            
        self.threshold = config_data.get('threshold', 1000)
        self.data_path = config_data.get('data_path', './data')
        self.categories = config_data.get('categories', ['chair', 'sofa', 'table', 'bed'])
        self.target_type = config_data.get('target_type', ['obj'])
        self.amount = config_data.get('amount', 500)
        self.seed = config_data.get('seed', random.randint(0, 1000))
        assert isinstance(self.threshold, float) and isinstance(self.data_path, str) \
            and isinstance(self.categories, list) and isinstance(self.amount, int) # check type
        assert set(self.target_type).issubset({'stl', 'obj'}) # supported target type
        logging.info('init fetcher, preparing model and processor')
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model, transform_clip = clip.load("ViT-B/32", device=self.device)
            self.clip = model
            self.transform_clip = transform_clip  
            model, transform_blip = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(self.device), AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip = model
            self.transform_blip = transform_blip  
        except Exception as e:
            logging.error('error when init fetcher, please check your network')
            logging.error(e)
            exit(2)
        logging.info('init fetcher success')

    def fetch(self, data_path, categories, amount):
        if not os.path.exists(data_path):
            logging.info('data path not exists, create it at '+data_path)
            os.mkdir(data_path)
        if not os.path.exists(os.path.join(data_path, 'fetched')):
            os.mkdir(os.path.join(data_path, 'fetched'))
        for category in categories:
            if not os.path.exists(os.path.join(data_path, 'fetched', category)):
                os.mkdir(os.path.join(data_path, 'fetched', category))
                    
        annotations = oxl.get_annotations(download_dir=data_path)
        annotations = annotations[annotations['fileType'].isin(self.target_type)]
        annotations = annotations.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        fetched_amount = 0
        for _, row in annotations.iterrows():
            info = 'source:'
            file_type = row['fileType']
            if file_type not in self.target_type:
                continue
            source, sha256 = row['source'], row['sha256']
            info = source
            info += ' sha256:'+sha256
            
            path_dict = oxl.download_objects(objects=pd.DataFrame(row).transpose(), download_dir='./data/')
            if len(path_dict) == 0:
                info += ' download failed'
                logging.warning(info)
                continue
            path = list(path_dict.values())[0]
            try:
                image, image_path = render[file_type](path)
            except Exception as e:
                info += ' render failed'
                logging.warning(info)
                logging.error(e)
                continue
                
            for category in categories:
                similarity, is_match = self.test_clip(image, category, self.threshold)
                is_match = is_match or self.test_clip_binary(image, category)
                is_match = is_match or self.test_blip(image, category)
                if is_match:
                    info += ' match '+category+' with similarity '+str(similarity)
                    fetched_amount += 1
                    shutil.move(image_path, os.path.join(data_path, 'fetched', category, sha256+'.'+image_path.split('.')[-1]))
                    shutil.move(path, os.path.join(data_path, 'fetched', category, sha256+'.'+file_type))
                    break
                else:
                    info += ' not match '+category
                        
                     
            if os.path.exists(path):
                os.remove(path)
            if os.path.exists(image_path):
                os.remove(image_path)
            
            logging.info(info)
            
            if fetched_amount >= amount:
                logging.info('fetched '+str(fetched_amount)+' files')
                return True
            
        logging.info('fetched '+str(fetched_amount)+' files')
        return False
            
    def test_clip(self, image, category, threshold):
        image = self.transform_clip(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(["a "+category]).to(self.device)
        with torch.no_grad():
            image_features = self.clip.encode_image(image)
            text_features = self.clip.encode_text(text)

        similarity_score = (image_features @ text_features.T).mean()

        return similarity_score, similarity_score > threshold
    
    def test_clip_binary(self, images, category, threshold=0):
        return self.test_clip(images, category, threshold)[0] > self.test_clip(images, 'object', threshold)[0] + 0.3

    def test_blip(self, image, category):
        inputs = self.transform_blip(image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.blip.generate(**inputs, max_new_tokens=20)
        generated_text = self.transform_blip.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return category in generated_text

    def __call__(self):
        if self.fetch(self.data_path, self.categories, self.amount):
            print("Done.")
        else:
            print("Not enough data, consider lowering the threshold or the amount.")
        return 0
        


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        filename='fetch.log',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)

    fetcher = Fetcher('fetch.json')
    fetcher()
    
