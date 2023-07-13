from PIL import Image, ImageFilter
from scipy.special import expit
import numpy as np
import cv2

def dilate(masks):
    mask_image = Image.fromarray(masks).convert("L")
    np_image = np.array(mask_image)
    kernel = np.ones((5, 5), np.uint8)

    # open_cv_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)

    img_dilation = cv2.dilate(np_image, kernel, iterations=50)
    # img_dilation = Image.fromarray(img_dilation)
    return img_dilation
    # white_image = Image.new('L', masks.T.shape, 255)

def process_masks(masks):
    # masks = masks.T
    width, height = masks.shape

    white_image = Image.new('L', masks.T.shape, 255)

    masks = dilate(masks)
    mask_image = Image.fromarray(masks).convert("L")
  

    blended_image = Image.blend(mask_image, white_image, alpha=0.1)


    # Áp dụng hàm sigmoid cho hình ảnh với blend img, beta = 10

    sigmoid_array = expit(np.array(blended_image)/255*10 - 1)
    sigmoid_image = Image.fromarray(sigmoid_array * 255).convert("L")

    # sigmoid_image = blended_image
    blurred_mask_image = sigmoid_image.filter(ImageFilter.GaussianBlur(radius=60))

    # Chuyển đổi lại hình ảnh PIL thành mảng np
    blurred_mask = np.array(blurred_mask_image) /255
   
    return blurred_mask

