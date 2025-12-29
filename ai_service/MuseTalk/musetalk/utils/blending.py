from PIL import Image
import numpy as np
import cv2
import copy


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    """
    对图像进行面部解析，生成面部区域的掩码。

    Args:
        image (PIL.Image): 输入图像。

    Returns:
        PIL.Image: 面部区域的掩码图像。
    """
    seg_image = fp(image, mode=mode)  # 使用 FaceParsing 模型解析面部
    if seg_image is None:
        print("error, no person_segment")  # 如果没有检测到面部，返回错误
        return None

    seg_image = seg_image.resize(image.size)  # 将掩码图像调整为输入图像的大小
    return seg_image


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.5, mode="raw", fp=None):
    """
    将裁剪的面部图像粘贴回原始图像，并进行一些处理。

    Args:
        image (numpy.ndarray): 原始图像（身体部分）。
        face (numpy.ndarray): 裁剪的面部图像。
        face_box (tuple): 面部边界框的坐标 (x, y, x1, y1)。
        upper_boundary_ratio (float): 用于控制面部区域的保留比例。
        expand (float): 扩展因子，用于放大裁剪框。
        mode: 融合mask构建方式 

    Returns:
        numpy.ndarray: 处理后的图像。
    """
    # 将 numpy 数组转换为 PIL 图像
    body = Image.fromarray(image[:, :, ::-1])  # 身体部分图像(整张图)
    face = Image.fromarray(face[:, :, ::-1])  # 面部图像

    x, y, x1, y1 = face_box  # 获取面部边界框的坐标
    crop_box, s = get_crop_box(face_box, expand)  # 计算扩展后的裁剪框
    x_s, y_s, x_e, y_e = crop_box  # 裁剪框的坐标
    face_position = (x, y)  # 面部在原始图像中的位置

    # 从身体图像中裁剪出扩展后的面部区域（下巴到边界有距离）
    face_large = body.crop(crop_box)
        
    ori_shape = face_large.size  # 裁剪后图像的原始尺寸

    # 对裁剪后的面部区域进行面部解析，生成掩码
    mask_image = face_seg(face_large, mode=mode, fp=fp)
    
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 裁剪出面部区域的掩码
    
    mask_image = Image.new('L', ori_shape, 0)  # 创建一个全黑的掩码图像
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 将面部掩码粘贴到全黑图像上
    
    
    # 保留面部区域的上半部分（用于控制说话区域）
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)  # 计算上半部分的边界
    modified_mask_image = Image.new('L', ori_shape, 0)  # 创建一个新的全黑掩码图像
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))  # 粘贴上半部分掩码
    
    
    # 对掩码进行高斯模糊，使边缘更平滑
    blur_kernel_size = int(0.05 * ori_shape[0] // 2 * 2) + 1  # 计算模糊核大小
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)  # 高斯模糊
    #mask_array = np.array(modified_mask_image)
    mask_image = Image.fromarray(mask_array)  # 将模糊后的掩码转换回 PIL 图像
    
    # 将裁剪的面部图像粘贴回扩展后的面部区域
    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    
    body.paste(face_large, crop_box[:2], mask_image)
    
    body = np.array(body)  # 将 PIL 图像转换回 numpy 数组

    return body[:, :, ::-1]  # 返回处理后的图像（BGR 转 RGB）


def get_image_blending(image, face, face_box, mask_array, crop_box):
    """
    Optimized Blending using pure OpenCV/Numpy (No PIL) for speed.
    """
    try:
        x, y, x1, y1 = face_box
        x_s, y_s, x_e, y_e = crop_box
        
        # Ensure mask is float 0-1
        if mask_array.dtype == np.uint8:
             mask = mask_array.astype(np.float32) / 255.0
        else:
             mask = mask_array
             
        # Expand dims for broadcasting if needed (H,W) -> (H,W,3)
        if len(mask.shape) == 2:
             mask = mask[..., None]
             
        # Extract Regions with Padding
        h_img, w_img = image.shape[:2]
        crop_h, crop_w = y_e - y_s, x_e - x_s
        
        # Initialize full canvas for background (matching mask size)
        face_large_roi = np.zeros((crop_h, crop_w, 3), dtype=np.float32)
        
        # Calculate intersection
        src_x, src_y = max(0, x_s), max(0, y_s)
        src_x2, src_y2 = min(w_img, x_e), min(h_img, y_e)
        
        dst_x, dst_y = src_x - x_s, src_y - y_s
        dst_x2, dst_y2 = dst_x + (src_x2 - src_x), dst_y + (src_y2 - src_y)
        
        # Copy valid image pixels to canvas
        if src_x2 > src_x and src_y2 > src_y:
            valid_pixels = image[src_y:src_y2, src_x:src_x2].astype(np.float32)
            face_large_roi[dst_y:dst_y2, dst_x:dst_x2] = valid_pixels
            
        face_small = face.astype(np.float32)
        
        # Calculate ROI in face_large where the small face goes
        # box coords (x,y) are relative to original image
        # crop coords (x_s, y_s) are relative to original image
        # relative coords:
        rx = x - x_s
        ry = y - y_s
        rx1 = x1 - x_s
        ry1 = y1 - y_s
        
        # Blending logic: out = face * mask + bg * (1-mask)
        # We need the mask to match the FACE ROI size (Wait, mask is FULL CROP SIZE? Yes)
        # mask is size (crop_h, crop_w)
        
        # Safely place face_small into source canvas
        source = face_large_roi.copy()
        
        # Ensure rx, ry fall within bounds (should be guaranteed by logic, but robust check)
        # rx, rx1 are relative to 0,0 of crop_box.
        # face_small fits inside because crop_box encompasses face_box.
        source[ry:ry1, rx:rx1] = face_small
        
        # Composite
        # out = source * mask + bg * (1-mask)
        # bg is face_large_roi (padded)
        
        out = source * mask + face_large_roi * (1.0 - mask)
        
        # Clip and Cast
        out = np.clip(out, 0, 255).astype(np.uint8)
        
        # Put back into original image
        # If we modify 'image' in place? 
        # Better to return copy to be safe, or just modify the roi in a copy of image
        # user expects 'body' returned.
        
        # Optimization: We only need to return the CROPPED region? 
        # run_inference writes 'combine_frame.tobytes()'.
        # 'combine_frame' is expected to be the FULL frame?
        # Yes, line 109 returns body.
        
        # Full frame copy
        result = image.copy()
        
        # Safe paste 'out' back into 'result' handling clipping AGAIN
        # out is size (h_s, w_s) matching crop_box size.
        # We need to paste it at x_s, y_s.
        
        h_img, w_img = image.shape[:2]
        
        # Valid roi in image
        x_start = max(0, x_s)
        y_start = max(0, y_s)
        x_end = min(w_img, x_e)
        y_end = min(h_img, y_e)
        
        # Corresponding roi in 'out'
        # out starts at x_s.
        # if x_s < 0, we need to skip the first -x_s pixels of 'out'
        out_x_start = x_start - x_s
        out_y_start = y_start - y_s
        out_x_end = out_x_start + (x_end - x_start)
        out_y_end = out_y_start + (y_end - y_start)
        
        if out_x_end > out_x_start and out_y_end > out_y_start:
             result[y_start:y_end, x_start:x_end] = out[out_y_start:out_y_end, out_x_start:out_x_end]
        
        return result
    except Exception as e:
        print(f"Blending Error: {e}")
        return image


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.5, fp=None, mode="raw"):
    body = Image.fromarray(image[:,:,::-1])
    # ... (Keep existing logic for PREPARE, it uses PIL which handles bounds)
    # But wait, we should optimize PREPARE too if it's slow?
    # PREPARE is called ONCE per session. Speed is less critical.
    # Blending is called PER FRAME.
    # So we leave PREPARE as is.
    
    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box
