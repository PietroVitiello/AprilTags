from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
# import torchvision.transforms as T

def open_image(path):
    return Image.open(path)

def open_image_numpy(path):
    image = Image.open(path)
    return np.array(image)

def draw_bounding_boxes(image, bbox):
    bbox = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
    img1 = ImageDraw.Draw(image)  
    img1.rectangle(bbox, outline ="red", width=4)

def draw_mask(image, mask, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.gca().imshow(mask_image)
    plt.axis('off')
    plt.show()