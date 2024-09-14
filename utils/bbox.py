import cv2
import torch
import os

def draw_bbox(imgs_path: str, bbox: torch.Tensor):
    """
    Only support single image input

    Args: 
        img_path: List[str | os.pathlike]
        bbox: torch.Tensor -> (B,4)
    """
    for index in range(len(imgs_path)):
        img_path = imgs_path[index]
        image_name = img_path.split('/')
        target_path = os.path.join(img_path.split('/')[:-1],"result_"+image_name)
        
        img = cv2.imread(img_path)
        x1,y1,x2,y2 = round(float(bbox[index,0])), round(float(bbox[index,1])), round(float(bbox[index,2])), round(float(bbox[index,3]))
        img = cv2.rectangle(img,(x1,y1,),(x2,y2),(0,0,255),2)
        cv2.imwrite(target_path, img,)
