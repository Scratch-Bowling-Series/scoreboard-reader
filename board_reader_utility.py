import json
import math
import os
import time
from difflib import SequenceMatcher

import torch
import base64
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt




##--------------------------------
## Colors
##--------------------------------
from shapely import Polygon

COLOR_BLACK = (0,0,0)
COLOR_RED = (0,0, 255)
COLOR_GREEN = (0,255, 0)
COLOR_BLUE = (255,0, 0)



##--------------------------------
## Board Object
##--------------------------------
def most_prominent_board(boards):
    detection_prominences = [board.detection_prominence for board in boards]
    max_detection_prominence_index = np.argmax(detection_prominences)
    prominent_board = boards[max_detection_prominence_index]
    return prominent_board



##--------------------------------
## Image
##--------------------------------
def image_shape(image):
    w = h = a = 0
    if image is not None and len(image) > 0:
        w, h = image.shape[:2]
        a = w * h
    return h, w, a

def image_valid(image):
    if image is not None and len(image) > 0:
        w, h = image.shape[:2]
        a = w * h
        if a:
            return True
    return False

def image_from_relative_path(relative_image_path):
    absolute_path = os.path.join(os.getcwd(), relative_image_path)
    if os.path.isfile(absolute_path):
        return cv.imread(absolute_path)

def images_from_relative_path(relative_folder_path):
    absolute_folder_path = os.path.join(os.getcwd(), relative_folder_path)
    images = {}
    for filename in os.listdir(absolute_folder_path):
        filepath = os.path.join(absolute_folder_path, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = remove_extension_from_file_name(filename)
            images[filename] = cv.imread(filepath)
    return images


##--------------------------------
## Files
##--------------------------------
def remove_extension_from_file_name(filename):
    return os.path.splitext(filename)[0] # v2 - modified by chatgpt - simplified code

def image_id_from_file_name(filename):
    numeric_chars = ''.join(filter(str.isdigit, filename))
    if numeric_chars:
        return int(numeric_chars)
    return None

def read_json_file(relative_file_path):
    absolute_path = os.path.join(os.getcwd(), relative_file_path)
    if os.path.isfile(absolute_path):
        with open(absolute_path, 'r') as file:
            data = json.load(file)
        return data
    return None

def write_json_file(relative_file_path, data):
    absolute_path = os.path.join(os.getcwd(), relative_file_path)
    with open(absolute_path, 'w') as file:
        json.dump(data, file)


##--------------------------------
## Image Modification
##--------------------------------
def convert_grayscale(image):
    if len(image.shape) == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        return image

def invert_grayscale_if_more_black(image, sensitivity=0.4):
    if len(image.shape) > 2:
        image = convert_grayscale(image)

    _, binary_image = cv.threshold(image, 128, 255, cv.THRESH_BINARY)

    ratio = np.sum(binary_image == 0) / (image.size + 1e-6)

    if ratio > sensitivity:
        image = invert_grayscale(image)

    return image

def invert_grayscale(image):
    if len(image.shape) > 2:
        image = convert_grayscale(image)
    return  cv.bitwise_not(image)

def rotate_image_by_angle(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    return cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

def crop_image_by_bounding_box(image, bounding_box):
    x, y, w, h = bounding_box
    return image[y:y+h, x:x+w]


def crop_image_by_polygon(image, polygon):

    # Convert polygon to numpy array
    polygon_np = np.array(polygon, dtype=np.int32)

    # Ensure polygon_np is 3-dimensional (1, n_points, 2)
    if polygon_np.ndim == 2:
        polygon_np = polygon_np.reshape((-1, 1, 2))

    # Create a single-channel mask with the polygon filled with white color (255)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    try:
        cv.fillPoly(mask, [polygon_np], 255)
    except Exception as e:
        raise

    # Bitwise AND operation to get the region of interest (ROI)
    result = cv.bitwise_and(image, image, mask=mask)

    # Find the bounding box of the polygon
    x, y, w, h = cv.boundingRect(polygon_np)

    # Crop the image based on the bounding box
    cropped_image = result[y:y + h, x:x + w]

    return cropped_image

def add_margin_to_image(image, margin_percentage=5, margin_color=(0, 0, 0)):
    height, width = image.shape[:2]
    margin_percentage *= 0.01
    margin_size_w = int(width * margin_percentage)
    margin_size_h = int(height * margin_percentage)
    return cv.copyMakeBorder(image, margin_size_h, margin_size_h, margin_size_w, margin_size_w, cv.BORDER_CONSTANT, value=margin_color)

def remove_borders_from_image(image, initial_margin=10, threshold=80):
    gray = convert_grayscale(image)

    # Get image dimensions
    height, width = gray.shape

    # Calculate the margin size based on the percentage
    margin_percent = initial_margin

    def should_remove_border(border_region, threshold_percent):
        # Check if the percentage of black pixels in the border region is above the threshold
        black_pixels_percentage = (np.count_nonzero(border_region == 0) / border_region.size) * 100
        return black_pixels_percentage > threshold_percent

    def set_border_to_white(margin_size, side):
        # Set the entire margin to white based on the specified side
        if side == 'top':
            image[:margin_size, :] = 255
        elif side == 'bottom':
            image[-margin_size:, :] = 255
        elif side == 'left':
            image[:, :margin_size] = 255
        elif side == 'right':
            image[:, -margin_size:] = 255

    def remove_border_from_side(side, margin_size):
        if side == 'top':
            top_border_region = gray[:margin_size, :]
            if should_remove_border(top_border_region, threshold):
                set_border_to_white(margin_size, 'top')
                return True
        elif side == 'bottom':
            bottom_border_region = gray[-margin_size:, :]
            if should_remove_border(bottom_border_region, threshold):
                set_border_to_white(margin_size, 'bottom')
                return True
        elif side == 'right':
            right_border_region = gray[:, -margin_size:]
            if should_remove_border(right_border_region, threshold):
                set_border_to_white(margin_size, 'right')
                return True
        elif side == 'left':
            left_border_region = gray[:, :margin_size]
            if should_remove_border(left_border_region, threshold):
                set_border_to_white(margin_size, 'left')
                return True
        return False

    removed_sides = []
    margin_size_x = int(width * margin_percent / 100)
    margin_size_y = int(height * margin_percent / 100)
    max_attempts = max(margin_size_y, margin_size_x)
    attempts = 0

    # Attempt to remove borders with the initial margin
    while len(removed_sides) < 4 and attempts < max_attempts:
        if margin_size_y > 1:
            if 'top' not in removed_sides and remove_border_from_side('top', margin_size_y):
                removed_sides.append('top')
            if 'bottom' not in removed_sides and remove_border_from_side('bottom', margin_size_y):
                removed_sides.append('bottom')
            margin_size_y -= 1
        if margin_size_x > 1:
            if 'right' not in removed_sides and remove_border_from_side('right', margin_size_x):
                removed_sides.append('right')
            if 'left' not in removed_sides and remove_border_from_side('left', margin_size_x):
                removed_sides.append('left')
            margin_size_x -= 1

        attempts += 1

    return image

def enhance_image_contrast(image, threshold_min=0, threshold_max=255, clipLimit=2.0, tileGridSize=(8, 8)):
    _, binary_image = cv.threshold(image, threshold_min, threshold_max, cv.THRESH_BINARY + cv.THRESH_OTSU)
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(binary_image)



##--------------------------------
## Serialization
##--------------------------------
def serialize_image_to_base64(image):
    _, img_encoded = cv.imencode('.png', image)
    image_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return image_base64

def unserialize_base64_image(encoded_image):
    if ',' in encoded_image:
        encoded_image = encoded_image.split(',')[1]
    padding = '=' * (-len(encoded_image) % 4)
    encoded_image += padding
    decoded_bytes = base64.b64decode(encoded_image)
    np_array = np.frombuffer(decoded_bytes, dtype=np.uint8)
    image = cv.imdecode(np_array, cv.IMREAD_COLOR)
    return image


##--------------------------------
## Lines
##--------------------------------
def detect_lines_from_image(image):
    lines = []

    canny_low = 100
    canny_high = 200
    hough_threshold = 50
    min_line_length = 100
    max_line_gap = 10
    contour_area_threshold = 100

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred = cv.GaussianBlur(thresholded, (5, 5), 0)
    canny = cv.Canny(blurred, canny_low, canny_high, apertureSize=3)
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv.contourArea(contour) > contour_area_threshold]
    contour_image = np.zeros_like(gray)
    cv.drawContours(contour_image, contours, -1, 255, thickness=cv.FILLED)
    result_canny = cv.bitwise_and(canny, contour_image)
    detected_lines = cv.HoughLinesP(result_canny, 1, np.pi / 180, threshold=hough_threshold,
                           minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines_valid(detected_lines):
        lines = [line[0] for line in detected_lines]

    return lines

def line_intersection_point(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
    return np.array([x, y], dtype=np.float32)

def filter_lines_by_radian(lines, radian):
    return [line for line in lines if abs(np.arctan2(line[3] - line[1], line[2] - line[0])) >= radian]

def filter_lines_by_angle(lines, min_angle, max_angle):
    return [line for line in lines if np.radians(min_angle) < abs(np.arctan2(line[3] - line[1], line[2] - line[0])) > np.radians(max_angle)]

def filter_lines_by_length(lines, min_length=-1, max_length=-1):
    return [line for line in lines if (min_length == -1 or line_length(line) >= min_length) and (max_length == -1 or line_length(line) <= max_length)]

def line_length(line):
    return abs(math.hypot(line[2] - line[0], line[3] - line[1]))

def average_angle_from_lines(lines):
    average_angle = 0
    angles = [abs(np.arctan2(line[3] - line[1], line[2] - line[0]) * 180 / np.pi) for line in lines]
    if angles:
        average_angle = np.mean(angles)
    return average_angle

def average_radian_from_lines(lines):
    average_radian = 0
    radians = [abs(np.arctan2(line[3] - line[1], line[2] - line[0])) for line in lines]
    if radians:
        average_radian = np.mean(radians)
    return average_radian

def median_angle_from_lines(lines):
    average_angle = 0
    angles = [abs(np.arctan2(line[3] - line[1], line[2] - line[0]) * 180 / np.pi) for line in lines]
    if angles:
        average_angle = np.median(angles)
    return average_angle

def median_radian_from_lines(lines):
    average_radian = 0
    radians = [abs(np.arctan2(line[3] - line[1], line[2] - line[0])) for line in lines]
    if radians:
        average_radian = np.median(radians)
    return average_radian

def lines_valid(lines):
    if lines is None or len(lines) == 0:
        return False
    return True

def rotation_matrix_from_lines(lines, image_shape):
    image_width, image_height = image_shape
    median_vertical_radian = median_radian_from_lines(filter_lines_by_angle(lines, min_angle=85, max_angle=95))
    median_horizontal_radian = median_radian_from_lines(filter_lines_by_angle(lines, min_angle=0, max_angle=5))
    rotation_matrix = cv.getRotationMatrix2D((image_width // 2, image_height // 2), median_horizontal_radian, 1.0)
    rotation_matrix[1, 2] = -median_vertical_radian
    return rotation_matrix



##--------------------------------
## Boxes (x1,y1,x2,y2)
##--------------------------------
def box_iou(box1, box2):
    if box1.dim() == 1:
        box1 = box1.unsqueeze(0)
    if box2.dim() == 1:
        box2 = box2.unsqueeze(0)

    if box1.size(-1) == 0 or box2.size(-1) == 0:
        return torch.zeros((box1.size(0), box2.size(0)), dtype=torch.float32, device=box1.device)

    (a1, a2), (b1, b2) = box1.unsqueeze(-2).chunk(2, -1), box2.unsqueeze(-3).chunk(2, -1)

    iw = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(min=0)
    ih = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(min=0)

    inter = iw * ih

    area1 = (a2 - a1).prod(-1)
    area2 = (b2 - b1).prod(-1)

    iou = inter / (area1 + area2 - inter + 1e-16)
    return iou



##--------------------------------
## Bounds (x,y,w,h)
##--------------------------------
def scale_bound_from_center(bound, scale_factor, min_x=0, min_y=0, max_x=99999, max_y=99999):
    x, y, x_max, y_max = bound

    # Calculate the center of the bounding box
    center_x = (x + x_max) / 2
    center_y = (y + y_max) / 2

    # Calculate the new half-width and half-height after scaling
    new_half_width = (x_max - x) * scale_factor / 2
    new_half_height = (y_max - y) * scale_factor / 2

    # Calculate the new coordinates based on the scaled half-width and half-height
    new_x = max(min_x, int(center_x - new_half_width))
    new_y = max(min_y, int(center_y - new_half_height))
    new_x_max = min(max_x, int(center_x + new_half_width))
    new_y_max = min(max_y, int(center_y + new_half_height))

    # Return the scaled bounds
    return [new_x, new_y, new_x_max, new_y_max]



##--------------------------------
## Polygons
##--------------------------------
def box_to_polygon(box):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def mask_to_polygon(mask):
    contour = largest_contour_from_mask(mask)
    return contour_to_polygon(contour)

def contour_to_polygon(contour):
    epsilon = 0.02 * cv.arcLength(contour, True)
    return cv.approxPolyDP(contour, epsilon, True)

def perspective_matrix_from_polygon(polygon):
    perspective_matrix = None
    bbox = None

    if len(polygon) >= 4:
        bbox = cv.boundingRect(polygon)
        src_points = polygon.reshape(-1, 2).astype(np.float32)
        src_points = src_points[np.argsort(np.sum(src_points, axis=1))]

        dst_points = np.float32(
            [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]],
             [bbox[0], bbox[1] + bbox[3]]])
        dst_points = dst_points[np.argsort(np.sum(dst_points, axis=1))]

        perspective_matrix = cv.getPerspectiveTransform(src_points, dst_points)


    return perspective_matrix, bbox

def polygon_from_overlap(poly1, poly2):
    shapely_poly1 = Polygon(poly1)
    shapely_poly2 = Polygon(poly2)

    # Check if polygons are valid
    if not shapely_poly1.is_valid or not shapely_poly2.is_valid:
        return None

    # Find the intersection (overlap) of the two polygons
    intersection = shapely_poly1.intersection(shapely_poly2)

    # If there's no intersection or it's not a polygon, return None
    if intersection.is_empty or not intersection.geom_type == 'Polygon':
        return None

    # Convert the intersection polygon back to numpy array format
    overlap_polygon = np.array(intersection.exterior.coords)

    return overlap_polygon

def polygon_iou(poly1, poly2):
    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    iou = intersection / union if union > 0 else 0.0

    return iou

def row_columns_to_polygons(row_polygon, column_polygons):
    polygons = []

    for column_polygon in column_polygons:

        iou = polygon_iou(row_polygon, column_polygon)
        # If there is overlap, create a polygon from the overlap
        if iou > 0:
            overlap_polygon = polygon_from_overlap(row_polygon, column_polygon)
            polygons.append(overlap_polygon)

    return polygons



##--------------------------------
## Masks
##--------------------------------
def resize_mask(mask, image_shape):
    image_width, image_height = image_shape
    return cv.resize(mask, (image_width, image_height))

def largest_contour_from_mask(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv.contourArea)


##--------------------------------
## Lists
##--------------------------------
def list_similarity_score(list1, list2):
    # Convert lists to strings for comparison
    str1 = '|'.join(map(str, list1))
    str2 = '|'.join(map(str, list2))

    # Calculate similarity score using SequenceMatcher
    matcher = SequenceMatcher(None, str1, str2)
    similarity_ratio = matcher.ratio()

    # Convert ratio to a 0-100 score
    similarity_score = int(similarity_ratio * 100)

    return similarity_score

def list_from_zip_match(list1, list2, match):
    return [item1 for item1, item2 in zip(list1, list2) if item2 == match]


##--------------------------------
## Timer
##--------------------------------
def start_timer():
    return Timer().start()

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        return self

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer not started")
        return round((time.time() - self.start_time) * 1000, 1)





##--------------------------------
## YOLO Ultralytics
##--------------------------------
def convert_yolo_mask_to_mask(yolo_mask):
    return np.uint8(yolo_mask.cpu().data.numpy().transpose(1, 2, 0))

def confidence_max(list, confidences):
    list_len = len(list)
    if list_len > 1:
        max_confidence_index = np.argmax(confidences)
        return list[max_confidence_index]
    elif list_len == 1:
        return list[0]



##--------------------------------
## Easy OCR Reader
##--------------------------------
def get_easyocr_extracted_text(result):
    return ' '.join([entry[1] for entry in result])



##--------------------------------
## Image Drawing
##--------------------------------
def draw_lines_to_image(image, lines, color=(255,255,255), thickness=5):
    for line in lines:
        x1, y1, x2, y2 = line
        cv.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def draw_points_to_image(image, points, color=(255,255,255), thickness=5):
    for point in points:
        x, y = point
        cv.circle(image, (int(x), int(y)), thickness, color, -1)

def draw_bounds_to_image(image, bounds, color=(255,255,255), thickness=5):
    for bound in bounds:
        x, y, w, h = bound
        cv.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)

def draw_boxes_to_image(image, boxes, color=(255,255,255), thickness=5):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def draw_polygons_to_image(image, polygons, color=(255,255,255), thickness=5):
    for polygon in polygons:
        cv.polylines(image, polygon, True, color, thickness)

def draw_perspective_matrix_to_image(image, matrix, color=(0, 255, 0), thickness=2):
    rows, cols = image.shape[:2]
    points = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    transformed_points = cv.perspectiveTransform(np.array([points]), matrix)[0]
    for i in range(4):
        pt1 = tuple(transformed_points[i])
        pt2 = tuple(transformed_points[(i + 1) % 4])
        cv.line(image, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), color, thickness)


##--------------------------------
## PLT Image Plot Class
##--------------------------------
class ImagePlot:

    def __init__(self):
        self.plot_images = []
        self.ncols = 2
        self.figsize = (8, 8)

    def plot(self, image, title=''):
        self.plot_images.append({'image': np.copy(image), 'title': title})

    def show(self):
        canvas = plt.figure().canvas
        canvas.required_interactive_framework = 'qt5agg'
        images = self.plot_images
        nrows = -(-len(images) // self.ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=self.ncols, figsize=self.figsize)
        axes = axes.flatten()

        for i, item in enumerate(images):
            image = item['image']
            title = item['title']

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            axes[i].imshow(image)
            axes[i].set_title(title, fontsize=10)
            axes[i].set_axis_off()

            for j in range(i + 1, nrows * self.ncols):
                axes[j].set_axis_off()

        plt.tight_layout()
        plt.show()




















