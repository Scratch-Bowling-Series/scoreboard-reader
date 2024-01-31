# Python Scoreboard Reader

## Overview
The Python Scoreboard Reader is an advanced, AI-powered system designed for the detection, segmentation, recognition, and post-processing of scoreboard images. It leverages state-of-the-art machine learning models and image processing techniques to deliver high accuracy in real-time scoreboard data extraction and analysis.

## Features
- **Board Detection**: Detects scoreboards in images using a YOLO-based model.
- **Board Segmentation**: Segments detected scoreboards into meaningful components like names, frames, and totals.
- **Board Recognition**: Recognizes and reads text from segmented scoreboard components using EasyOCR.
- **Data Post-Processing**: Cleans and structures recognized data for better readability and usability.
- **Robust Error Handling**: Well-defined exception handling for robust application performance.

## Demo
Check out the live demo [here](https://board-reader.bowl.sbs/).

## Prerequisites
- Python 3.6+
- OpenCV (cv)
- Numpy
- PyTorch
- EasyOCR
- Ultralytics YOLO

## Installation
1. **Clone the repository**
   ```git clone https://github.com/Scratch-Bowling-Series/scoreboard-reader.git```
2. **Navigate to the project directory**
```cd scoreboard-reader```
3. **Install dependencies**
```pip install -r requirements.txt```
## Usage
2. **Invoke the `read_image` function**
- You can use the `read_image` function to process your scoreboard images.
- Parameters:
  - `image`: Input image containing the scoreboard.
  - `single_board`: Boolean flag to indicate whether to process a single board or multiple.
  - `detection_model`, `segmentation_model`, `recognition_model`: Paths to the respective model weights.

## Model Weights

Below are the two versions for each of the three model weights used in the Python Scoreboard Reader:

| Model Type         | Version 1                                                                  | Version 2                                                                  |
|--------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Detection Model    | [`detection_v1.pt`](path_to_weights_folder/detection_v1.pt)                | [`detection_v2.pt`](path_to_weights_folder/detection_v2.pt)                |
| Segmentation Model | [`segmentation_v1.pt`](path_to_weights_folder/segmentation_v1.pt)          | [`segmentation_v2.pt`](path_to_weights_folder/segmentation_v2.pt)          |
| Recognition Model  | [`recognition_v1.pt`](path_to_weights_folder/recognition_v1.pt)            | [`recognition_v2.pt`](path_to_weights_folder/recognition_v2.pt)            |

Ensure you use the correct version according to your specific needs and compatibility with the rest of your system.

## Board Object Class Structure

The `Board` class represents a detected scoreboard and encapsulates all relevant data obtained during the processing stages. Below is an overview of its structure and the role of each attribute:

### Attributes
- `id`: A unique identifier for each detected board.
- `image`: The image of the detected board after perspective transformation and pre-processing.
- `detection_prominence`: A score indicating the prominence of the board in the image.
- `detection_confidence`: The confidence score of the detection model for the detected board.
- `recognition_confidence`: The confidence score for the recognized text within the board.
- `name_images`: A list of cropped images of player names.
- `frame_images`: A list of cropped images for each frame of the scoreboard.
- `total_images`: A list of cropped images of the total scores.
- `segmented_rows`: Segmented parts of the board, each representing a player's row.
- `segmented_lane`: The segmented part of the board representing the lane information.
- `recognition_rows`: Recognized data for each row in the board.
- `recognition_lane`: Recognized data for the lane.
- `lane_number`: Processed lane number information.
- `bowler_count`: The number of players (bowlers) detected in the scoreboard.
- `active_bowler_index`: Index of the bowler currently playing.
- `winner_bowler_index`: Index of the bowler with the highest score.
- `current_frame`: The current frame being played.
- `bowler_names`: List of names of the bowlers.
- `bowler_totals`: List of total scores of the bowlers.
- `bowler_frames`: Frame-by-frame scores for each bowler.
- `times`: A dictionary containing performance metrics like processing times for different stages.

# Python Scoreboard Reader - Flask Application

## Overview
The Python Scoreboard Reader Flask application provides a web interface for users to upload scoreboard images and receive processed results. The app uses the core functionality of the Python Scoreboard Reader to detect, segment, recognize, and post-process the scoreboards within the images.

## Features
- **Web Interface**: A user-friendly web interface for uploading scoreboard images.
- **Single or Multiple Board Processing**: Supports processing of single or multiple scoreboards within an image.
- **Real-Time Results**: Processes uploaded images in real-time and displays the recognized scoreboard data.
- **Error Handling**: Robust error handling for a smoother user experience.

## Setup and Running the Application
1. **Ensure Flask is installed**
   If Flask is not installed, you can install it using pip:
```pip install Flask```

2. **Start the Flask application**
Navigate to the project directory and run:
```python flask_app.py```

## Usage
1. **Access the Web Interface**
- Open your web browser and navigate to `http://localhost:5000/` (or the host and port you specified).

2. **Upload an Image**
- Use the web interface to upload a scoreboard image. The system supports processing of single or multiple scoreboards within the image.

3. **View Results**
- After processing, the recognized scoreboard data will be displayed on the web page.

## Endpoints
- **`/` (GET)**: Home page of the web application.
- **`/read-image/` (POST)**: Endpoint for processing uploaded images. Expects a JSON with `single_board` flag and `image` as a base64 encoded string.

## Error Handling
- The application provides descriptive error messages in case of invalid inputs or processing errors, ensuring a smooth user experience.

## Customization
- You can customize the Flask application settings such as host, port, and debug mode through command-line arguments when starting the app.

This Flask application serves as a convenient interface for interacting with the Python Scoreboard Reader, making it accessible for users without the need for direct interaction with the Python code.

## Contributing
Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Christian Starr - christian@scratchbowling.com

Project Link: [https://github.com/Scratch-Bowling-Series/scoreboard-reader](https://github.com/Scratch-Bowling-Series/scoreboard-reader)
