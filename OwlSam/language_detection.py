import requests
from PIL import Image
import torch
import torch.nn as nn

from ChatGPT import ChatGPT
from OwlViT import OWL_ViT

class LanguageDetector():

    def __init__(self) -> None:
        self.chatgpt = ChatGPT()
        self.owl = OWL_ViT()

    def __call__(self, image, task_desc):
        object_name = self.chatgpt(task_desc)
        print(object_name)
        data = {
        "texts": [object_name],
        "image": image,
        }
        return self.owl(data)
    
if __name__ == "__main__":
    from utils import draw_bounding_boxes, open_image
    import sys

    ld = LanguageDetector()
    image = open_image("/home/pita/Downloads/MicrosoftTeams-image.png")
    image.show()

    bbox = ld(image, sys.argv[1])

    draw_bounding_boxes(image, bbox)
    image.show()