import torch
import numpy as np
from segment_anything import (
    sam_model_registry, 
    SamPredictor, 
    SamAutomaticMaskGenerator
)
from PIL import Image

class SAM():
    def __init__(self, device = "cuda"):
        torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        model_type = 'vit_h'
        checkpoint = 'sam_vit_h_4b8939.pth'
        model = sam_model_registry[model_type](checkpoint = checkpoint)
        model.cuda(device)

        self.predictor = SamPredictor(model)
        self.mask_generator = SamAutomaticMaskGenerator(model)
    
    def set_image(self, image):        
        self.predictor.set_image(np.array(image)) # load the image to predictor


    def check_cuda(self):
        self.cuda_available = next(self.model.parameters()).is_cuda
        self.device = next(self.model.parameters()).get_device()
        if self.cuda_available:
            print ('Cuda is available.')
            print ('Device is {}'.format(self.device))
        else:
            print ('Cuda is not available.')
            print ('Device is {}'.format(self.device))

    def get_image_crop(self, image, masks, crop_mode = "w_bg"):
        # masks = self.infer_masks(input_point)
        masked_image = self.crop_by_masks(image, masks, crop_mode = crop_mode)
        size = max(masks.shape[0], masks.shape[1])
        left, top, right, bottom = self.seg_to_box(masks, size) # calculating the position of the top-left and bottom-right corners in the image
        
        image_crop = masked_image.crop((left * size, top * size, right * size, bottom * size)) # crop the image
        return image_crop

    
    
    def infer_masks(self, input_point = [[1800, 950]]):
        # input_point is a Nx2 array of point prompts to the model. Each point is in (X,Y) in pixels.
        input_label = [1]           
        # A length N array of labels for the point prompts. 1 indicates a foreground point and 0 indicates a background point.
        input_point = np.array(input_point)
        input_label = np.array(input_label)
        masks, scores, logits = self.predictor.predict(
            point_coords = input_point, 
            point_labels = input_label
        )
        masks = masks[0, ...]
        return masks

    def crop_by_masks(self, image, masks, crop_mode = "w_bg"):
        if crop_mode == "wo_bg":
            masked_image = image * masks[:,:,np.newaxis] + (1 - masks[:,:,np.newaxis]) * 255
            masked_image = np.uint8(masked_image)
        else:
            masked_image = np.array(image)

        masked_image = Image.fromarray(masked_image)
        return masked_image

    def boundary(self, inputs):
        col = inputs.shape[1]
        inputs = inputs.reshape(-1)
        lens = len(inputs)
        start = np.argmax(inputs)
        end = lens - 1 - np.argmax(np.flip(inputs))
        top = start // col
        bottom = end // col

        return top, bottom

    def seg_to_box(self, seg_mask, size):
        top, bottom = self.boundary(seg_mask)
        left, right = self.boundary(seg_mask.T)
        left, top, right, bottom = left / size, top / size, right / size, bottom / size # we normalize the size of boundary to 0 ~ 1

        return [left, top, right, bottom]
