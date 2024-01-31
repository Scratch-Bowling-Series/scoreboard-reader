import os
import random
import re
import sys

import cv2 as cv
import numpy as np
import torch
from easyocr import easyocr
from ultralytics import YOLO
torch.device("cpu")
import board_reader_utility as u

directory = os.path.dirname(__file__)
DETECTION_MODEL_DEFAULT = os.path.join(directory, 'weights/detection/best.pt')
SEGMENTATION_MODEL_DEFAULT = os.path.join(directory, 'weights/segmentation/best.pt')
RECOGNITION_MODEL_DEFAULT = os.path.join(directory, 'weights/recognition/best.pt')


##--------------------------------
## Methods
##--------------------------------
def read_image(image, single_board=True, detection_model=DETECTION_MODEL_DEFAULT,segmentation_model=SEGMENTATION_MODEL_DEFAULT,recognition_model=RECOGNITION_MODEL_DEFAULT):
    try:
        # Check if the image provided is valid
        if not u.image_valid(image):
            raise ImageInvalid('The image provided is invalid')

        # Initialize the classes needed for scoreboard recognition
        board_detection = BoardDetection(board_detection_model_path=detection_model)
        board_segmentation = BoardSegmentation(board_segmentation_model_path=segmentation_model)
        board_recognition = BoardRecognition(board_recognition_model_path=recognition_model)
        board_post_process = BoardPostProcess()

        # Use the board detection class to define the board objects in the image
        boards = board_detection.detect_boards(image)


        if single_board:
            # Retrieve the most prominent board object
            board = u.most_prominent_board(boards)

            # If we do not have a board, raise the board not found exception
            if len(boards) == 0:
                raise BoardNotFound('A board could not be detected in the image.')

            # Use the board segmentation class to populate the detail bounds for the board
            board_segmentation.segment_board(board)

            # Use the board recognition class to perform ocr on the individual detail bounds
            board_recognition.read_board(board)

            # Use the board post process class to clean up the recognized data and make predictions
            board_post_process.post_process(board)

            # Finally, return the detected board
            return board
        else:
            for board in boards:
                # Use the board segmentation class to populate the detail bounds for the board
                board_segmentation.segment_board(board)

                # Use the board recognition class to perform ocr on the individual detail bounds
                board_recognition.read_board(board)

                # Use the board post process class to clean up the recognized data and make predictions
                board_post_process.post_process(board)

            # Finally, return the detected boards
            return boards

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise e


##--------------------------------
## Board Class
##--------------------------------
class Board:
    
    id = 0
    image = None

    # Scores
    detection_prominence = 0
    detection_confidence = 0
    recognition_confidence = 0.0

    # Image Segmentations
    name_images = []
    frame_images = []
    total_images = []
    segmented_rows = []
    segmented_lane = None

    # Raw Recognized Data
    recognition_rows = []
    recognition_lane = ''

    # Processed Meta Data
    lane_number = ''
    bowler_count = 0
    active_bowler_index = -1
    winner_bowler_index = -1
    current_frame = 0
    bowler_names = []
    bowler_totals = []
    bowler_frames = []

    # Performance
    times = {}


    def __init__(self, id=0):
        self.id = id
        

    def serialize(self):
        serialized_board = {}
        try:
            def serialize_structure(k, x):
                if isinstance(x, list):
                    return [serialize_structure(k, item) for item in x]
                elif isinstance(x, dict):
                    return x
                elif k and 'image' in k:
                    return u.serialize_image_to_base64(x)
                elif isinstance(x, np.ndarray):
                    return x.tolist()
                elif hasattr(x, 'serialize') and callable(getattr(x, 'serialize')):
                    # return x.serialize()
                    return x
                else:
                    return x


            for attr in dir(self):
                if not callable(getattr(self, attr)) and not attr.startswith('__'):
                    serialized_board[attr] = serialize_structure(attr, getattr(self, attr))

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        finally:

            return serialized_board



##--------------------------------
## Board Detection Class
##--------------------------------
class BoardDetection:

    def __init__(self, board_detection_model_path, board_class_id=0):
        self.board_detection_model = YOLO(board_detection_model_path)
        #self.board_detection_model.to('cuda')
        self.board_class_id = board_class_id

    def detect_boards(self, image):
        try:
            # Start Timer
            timer = u.start_timer()

            image_height, image_width, image_area = u.image_shape(image)
            boards = []

            # Board inference
            results = self.board_detection_model.predict(source=image)

            inference_time = timer.stop()

            for result in results:
                confidences = result.boxes.conf.tolist()
                classes = result.boxes.cls.tolist()
                masks = result.masks

                for i, tup in enumerate(zip(classes, confidences, masks)):
                    cls, confidence, mask = tup

                    # Start Timer
                    timer = u.start_timer()

                    # Create the board object and board image
                    board = Board(id=len(boards))
                    board.image = np.copy(image)
                    board_copy = np.copy(image)

                    # Get the result mask and calculate the perspective matrix
                    mask = u.resize_mask(u.convert_yolo_mask_to_mask(mask),(image_height,image_width))
                    polygon = u.mask_to_polygon(mask)
                    perspective_matrix, bounding_box = u.perspective_matrix_from_polygon(polygon)

                    # Warp the board image using the perspective matrix, then crop from its source point bounding box
                    board.image = cv.warpPerspective(board.image, perspective_matrix, (image_height, image_width))
                    board.image = u.crop_image_by_bounding_box(board.image, bounding_box)

                    # Get the new board image height, width, and area after crop
                    board_image_height, board_image_width, board_image_area = u.image_shape(board.image)

                    # Use line detection to rectify any remaining distortion
                    #lines = u.detect_lines_from_image(board.image)
                    #rotation_matrix = u.rotation_matrix_from_lines(lines, (board_image_width, board_image_height))
                    #board.image = cv.warpAffine(board.image, rotation_matrix, (board_image_height, board_image_width))

                    # Add a black margin to the image
                    board.image = u.add_margin_to_image(board.image, 5, u.COLOR_BLACK)

                    # Calculate the board prominence score and retrieve the result confidence
                    board.detection_prominence = min(1, (board_image_area / image_area) * 1)
                    board.detection_confidence = confidence
                    board.times['detection_inference'] = inference_time
                    board.times['detection_post_process'] = timer.stop()
                    boards.append(board)


            return boards
        except Exception as e:
            raise BoardDetectionError(e)


##--------------------------------
## Board Segmentation Class
##--------------------------------
class BoardSegmentation:

    def __init__(self, board_segmentation_model_path):
        self.board_segmentation_model = YOLO(board_segmentation_model_path)
        #self.board_segmentation_model.to('cuda')

        self.CLASS_ACTIVE_ROW = 0
        self.CLASS_FRAME_COLUMN = 1
        self.CLASS_FRAME_TOTAL_ROW = 2
        self.CLASS_LANE = 3
        self.CLASS_NAME = 4
        self.CLASS_NAME_COLUMN = 5
        self.CLASS_PLAYER_ROW = 6
        self.CLASS_TOTAL_COLUMN = 7

    def segment_board(self, board, use_masks=False):
        try:
            # Start Timer
            timer = u.start_timer()

            image_width, image_height = u.image_shape(board.image)[:2]

            segmented_rows = []

            results = self.board_segmentation_model.predict(source=board.image)
            result = results[0]

            inference_time = timer.stop()

            boxes = result.boxes.xyxy.tolist()
            classes = result.boxes.cls.tolist()
            confidences = result.boxes.conf.tolist()

            if result.masks is not None and use_masks:
                masks = [u.resize_mask(u.convert_yolo_mask_to_mask(mask), (image_width, image_height)) for mask in result.masks]
            else:
                masks = []

            if use_masks:
                row_player_polygons = [u.mask_to_polygon(mask) for mask in u.list_from_zip_match(masks, classes, self.CLASS_PLAYER_ROW)]
                col_name_polygons = [u.mask_to_polygon(mask) for mask in u.list_from_zip_match(masks, classes, self.CLASS_NAME_COLUMN)]
                col_frame_polygons = [u.mask_to_polygon(mask) for mask in u.list_from_zip_match(masks, classes, self.CLASS_FRAME_COLUMN)]
                col_total_polygons = [u.mask_to_polygon(mask) for mask in u.list_from_zip_match(masks, classes, self.CLASS_TOTAL_COLUMN)]
                aoi_lane_polygons = [u.mask_to_polygon(mask) for mask in u.list_from_zip_match(masks, classes, self.CLASS_LANE)]
            else:
                row_player_polygons = [u.box_to_polygon(box) for box in u.list_from_zip_match(boxes, classes, self.CLASS_PLAYER_ROW)]
                col_name_polygons = [u.box_to_polygon(box) for box in  u.list_from_zip_match(boxes, classes, self.CLASS_NAME_COLUMN)]
                col_frame_polygons = [u.box_to_polygon(box) for box in  u.list_from_zip_match(boxes, classes, self.CLASS_FRAME_COLUMN)]
                col_total_polygons = [u.box_to_polygon(box) for box in  u.list_from_zip_match(boxes, classes, self.CLASS_TOTAL_COLUMN)]
                aoi_lane_polygons = [u.box_to_polygon(box) for box in  u.list_from_zip_match(boxes, classes, self.CLASS_LANE)]

            if len(col_name_polygons) > 1:
                col_total_polygons = [
                    u.confidence_max(col_total_polygons, u.list_from_zip_match(confidences, classes, self.CLASS_NAME_COLUMN))]

            if len(col_total_polygons) > 1:
                col_total_polygons = [
                    u.confidence_max(col_total_polygons, u.list_from_zip_match(confidences, classes, self.CLASS_TOTAL_COLUMN))]

            if len(aoi_lane_polygons) > 1:
                aoi_lane_polygons = u.confidence_max(col_total_polygons, u.list_from_zip_match(confidences, classes, self.CLASS_LANE))

            for row_player_polygon in row_player_polygons:
                segmented_rows.append([
                    row_player_polygon,
                    u.row_columns_to_polygons(row_player_polygon, col_name_polygons),
                    u.row_columns_to_polygons(row_player_polygon, col_frame_polygons),
                    u.row_columns_to_polygons(row_player_polygon, col_total_polygons),
                ])

            board.segmented_rows = segmented_rows
            board.segmented_lane = aoi_lane_polygons
            board.times['segmentation_inference'] = inference_time
            board.times['segmentation_calculation'] = timer.stop()
        except Exception as e:
            raise BoardSegmentationError(e)


##--------------------------------
## Board Recognition Class
##--------------------------------
class BoardRecognition:

    def __init__(self, board_recognition_model_path):
        self.board_recognition_model = easyocr.Reader(["en"])
        self.name_allow_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        self.frame_allow_list = ['0','1','2','3','4','5','6','7','8','9','x','/','-']
        self.total_allow_list = ['0','1','2','3','4','5','6','7','8','9']
        self.lane_allow_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']

    def read_board(self, board):

        # Start Timer
        timer = u.start_timer()

        recognition_rows = []

        for segmented_row in board.segmented_rows:
            name_image = None
            frame_images = []
            total_image = None

            name_polygons = segmented_row[1]
            frame_polygons = segmented_row[2]
            total_polygons = segmented_row[3]

            if len(name_polygons) > 0 and len(name_polygons[0]):
                name_image = u.crop_image_by_polygon(board.image, name_polygons[0])

            if len(total_polygons) > 0 and len(total_polygons[0]):
                total_image = u.crop_image_by_polygon(board.image, total_polygons[0])

            for frame_polygon in frame_polygons:
                if len(frame_polygon):
                    frame_images.append(u.crop_image_by_polygon(board.image, frame_polygon))

            name_text = self.read_roi(name_image, self.name_allow_list)
            total_text = self.read_roi(total_image, self.total_allow_list)
            frame_text = [self.read_roi(frame_image, self.frame_allow_list) for frame_image in frame_images]

            recognition_rows.append([
                name_text,
                frame_text,
                total_text
            ])

        lane_text = ''
        lane_polygon = board.segmented_lane
        lane_image = None

        if len(lane_polygon):
            lane_image = u.crop_image_by_polygon(board.image, lane_polygon)

        if u.image_valid(lane_image):
            lane_text = self.read_roi(lane_image, self.lane_allow_list)

        board.recognition_rows = recognition_rows
        board.recognition_lane = lane_text

        board.times['recognition_inference'] = timer.stop()

    def read_roi(self, image, allow_list, text_threshold=0.5):
        text = ''
        if u.image_valid(image):
            grayscale_image = u.convert_grayscale(image)
            contrast_enhanced = u.enhance_image_contrast(grayscale_image)
            flipped_image = u.invert_grayscale_if_more_black(contrast_enhanced)
            borders_removed = u.remove_borders_from_image(flipped_image)

            if len(borders_removed.shape) == 2:
                borders_removed = cv.cvtColor(borders_removed, cv.COLOR_GRAY2BGR)

            if borders_removed.dtype != np.uint8:
                borders_removed = borders_removed.astype(np.uint8)

            if u.image_valid(borders_removed):
                result = self.board_recognition_model.readtext(image=borders_removed, allowlist=allow_list, text_threshold=text_threshold)
                text = u.get_easyocr_extracted_text(result)
        return text



##--------------------------------
## Board Data Post Process Class
##--------------------------------
class BoardPostProcess:

    def __init__(self):
        pass
        

    def post_process(self, board):
        # Start Timer
        timer = u.start_timer()

        board.bowler_names = []
        board.bowler_frames = []
        board.bowler_totals = []

        for row in board.recognition_rows:
            name, frames, total = row
            
            name = self.post_process__name(name)
            frames = self.post_process__frames(frames)
            total = self.post_process__total(total)
            
            for frame_index, frame in enumerate(frames):
                if len(frame) > 0 and frame[len(frame)-1]:
                    board.current_frame = frame_index

            board.bowler_names.append(name)
            board.bowler_frames.append(frames)
            board.bowler_totals.append(total)

        board.bowler_count = len(board.bowler_names)
        board.active_bowler_index = board.bowler_count - 1
        board.lane_number = self.post_process__lane_number(board.recognition_lane)

        board.times['recognition_post_process'] = timer.stop()

    def post_process__frames(self, frames):
        processed_frames = []
        frame_pattern = re.compile(r"(X|\/|\-|\||\d+)")

        for frame_index, frame in enumerate(frames):

            # Find all parts of the frame using regex
            parts = frame_pattern.findall(frame)

            if len(parts) == 1:
                # Single part - could be a strike, a total after a spare, or the first throw
                part = parts[0]
                first, second, third, total = self.process_single_part_frame(part, frame_index, processed_frames)
            elif len(parts) == 2:
                # Two parts - could be first and second throws, or a throw and total score
                first, second, third, total = self.process_two_part_frame(parts, frame_index, processed_frames)
            elif len(parts) == 3:
                # Two parts - could be first and second throws, or a throw and total score
                first, second, third, total = self.process_three_part_frame(parts, frame_index, processed_frames)
            elif len(parts) == 4:
                # Three parts - likely first throw, second throw, and total score
                first, second, third, total = parts
            else:
                # Handle frames with more than three parts or other anomalies
                first, second, third, total = self.handle_anomalous_frame(parts, frame_index, processed_frames)

            if frame_index == 9:
                processed_frame = [first, second, third, total]
            else:
                processed_frame = [first, second, total]

            processed_frames.append(processed_frame)

        return processed_frames

    def process_single_part_frame(self, part, frame_index, processed_frames):
        first = second = third = total = 0
        # Implement logic for frames with a single part
        # Analyze previous frames, apply bowling rules, and make predictions
        # Return first, second, and total
        total = part

        return first, second, third, total

    def process_two_part_frame(self, parts, frame_index, processed_frames):
        first = second = third = total = 0
        # Implement logic for frames with two parts
        # Analyze previous frames, apply bowling rules, and make predictions
        # Return first, second, and total
        first = parts[0]
        total = parts[1]

        return first, second, third, total

    def process_three_part_frame(self, parts, frame_index, processed_frames):
        first = second = third = total = 0
        # Implement logic for frames with two parts
        # Analyze previous frames, apply bowling rules, and make predictions
        # Return first, second, and total
        first = parts[0]
        second = parts[1]
        total = parts[2]

        return first, second, third, total

    def handle_anomalous_frame(self, parts, frame_index, processed_frames):
        first = second = third = total = 0
        # Implement logic for handling frames that don't fit the usual patterns
        # This might involve complex predictions and handling a wide variety of edge cases
        # Return first, second, and total
        total = parts

        return first, second, third, total


    def post_process__name(self, name):
       return name

    def post_process__total(self, total):
        return total

    def post_process__lane_number(self, lane):
        return lane



##--------------------------------
## Exceptions
##--------------------------------
class ImageInvalid(Exception):
    pass
class BoardNotFound(Exception):
    pass
class BoardDetectionError(Exception):
    pass
class BoardSegmentationError(Exception):
    pass
class BoardRecogntitionError(Exception):
    pass