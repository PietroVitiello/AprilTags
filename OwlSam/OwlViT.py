import requests
from PIL import Image
import torch
import torch.nn as nn
import numpy as np

from transformers import OwlViTProcessor, OwlViTForObjectDetection

class OWL_ViT(nn.Module):

    def __init__(self, device: str ='cpu') -> None:
        super().__init__()

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.model.to(device=device)
        self.model.eval()
        self.device = device

    def get_complete_results(self, x: dict):
        texts = [x["texts"]]
        inputs = self.processor(text=texts, images=x["image"], return_tensors="pt")
        outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        # print(outputs)

        target_sizes = torch.Tensor([image.size[::-1]])

        # Convert outputs (bounding boxes and class logits) to COCO API
        return self.processor.post_process(outputs=outputs, target_sizes=target_sizes)

    # def forward(self, x: dict):
    #     if len(x["images"].shape) == 3:
    #         x["images"] = np.expand_dims(x["images"], 0)

    #     # print(x["images"].shape)

    #     bs, h, w, _ = x["images"].shape
    #     texts = [x["texts"]] * bs

    #     inputs = self.processor(text=texts, images=x["images"], return_tensors="pt")
    #     inputs.to(device=self.device)
    #     outputs = self.model(**inputs)
    #     target_sizes = (torch.ones((bs, 2)) * torch.tensor((h, w))[None]).to(device=self.device)

    #     # Convert outputs (bounding boxes and class logits) to COCO API
    #     results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)
    #     best_bboxes = torch.zeros((bs, len(texts[0]), 4), dtype=torch.int16, device=self.device)
        
    #     for object_id in range(len(x["texts"])):
    #         all_boxes, all_scores, all_labels = results[object_id]["boxes"].detach(), results[object_id]["scores"].detach(), results[object_id]["labels"].detach()
    #         for image_id in range(bs):
    #             boxes = all_boxes[all_labels==image_id]
    #             scores = all_scores[all_labels==image_id]

    #             best_choice = torch.argmax(scores)
    #             best_bboxes[image_id, object_id, :] = torch.round(boxes[best_choice])

    #     if len(x["texts"]) == 1:
    #         return best_bboxes[0]
    #     return best_bboxes
    
    def forward(self, x: dict):
        if len(x["images"].shape) == 3:
            x["images"] = np.expand_dims(x["images"], 0)

        bs, h, w, _ = x["images"].shape
        texts = [x["texts"]] * bs

        inputs = self.processor(text=texts, images=x["images"], return_tensors="pt")
        inputs.to(device=self.device)
        outputs = self.model(**inputs)
        target_sizes = (torch.ones((bs, 2)) * torch.tensor((h, w))[None]).to(device=self.device)

        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)[0]
        all_boxes, all_scores, all_labels = results["boxes"].detach(), results["scores"].detach(), results["labels"].detach()

        best_bboxes = torch.zeros((bs, len(texts[0]), 4), dtype=torch.int16, device=self.device)

        for object_id in range(len(x["texts"])):
            boxes = all_boxes[all_labels==object_id]
            scores = all_scores[all_labels==object_id]

            best_choice = torch.argmax(scores)
            best_bboxes[object_id,:] = torch.round(boxes[best_choice])

        if len(x["texts"]) == 1:
            return best_bboxes[0]
        return best_bboxes


if __name__ == "__main__":
    from utils import draw_bounding_boxes, open_image
    import sys

    if len(sys.argv) > 1:

        image_number = sys.argv[1]
        # image_dir = "/home/pita/Documents/PhD/OwlViT/fixed_present/head_camera_rgb_"
        image_dir = "/home/pita/Documents/Projects/OwlViT/fixed_present/head_camera_rgb_"
        image_dir += f"{image_number}.png"
        image = open_image(image_dir)

        text = [sys.argv[2]]

        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        # text = ["cat", "a photo of a remote"]

        owl = OWL_ViT()
        x = {
            "texts": text,
            "images": np.array(image),
        }

        # results = owl.get_complete_results(x)

        # print(results)

        # i = 0  # Retrieve predictions for the first image for the corresponding text queries
        # boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        # score_threshold = 0.1
        # for box, score, label in zip(boxes, scores, labels):
        #     box = [round(i, 2) for i in box.tolist()]
        #     if score >= score_threshold:
        #         print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

        bboxes = owl(x)

        draw_bounding_boxes(image, bboxes)
        image.show()
    else:
        owl = OWL_ViT()