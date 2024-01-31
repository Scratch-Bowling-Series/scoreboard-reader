import argparse
import json
import os
import sys

from flask import Flask, render_template, request, send_from_directory

import board_reader_utility as u
import board_reader as board_reader

app = Flask(__name__, template_folder='templates')

@app.route("/")
def home():
    data = {}
    return render_template('index.html', data=data)

@app.get('/test-image/<path:path>')
def send_image(path):
    return send_from_directory(
        directory='tests/test_images/', path=path
    )

@app.route("/read-image/", methods=['POST'])
def read_image():
    response = {'success': False}
    try:
        single_board = request.json.get('single_board', True)
        encoded_image = request.json.get('image')

        if not encoded_image:
            response['error'] = '/read-image/ requires a base64 encoded image in the body JSON.'
            return
        image = u.unserialize_base64_image(encoded_image)
        if not u.image_valid(image):
            response['error'] = 'The image is not valid.'
            return

        if single_board:
            board = board_reader.read_image(image)
            print(board.serialize)
            board = board.serialize()
            response['boards'] = [board]
        else:
            response['boards'] = [board.serialize() for board in board_reader.read_image(image, single_board=single_board)]

        response['success'] = True

    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        response['error'] = f"An exception of type {type(e).__name__} occurred: {str(e)}"
    finally:
        return json.dumps(response)


def parse_args(argv=None):
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--host",
                    type=str,
                    default='0.0.0.0',
                    help="Flask App Host IP")
    PARSER.add_argument("--port",
                    type=int,
                    default=5000,
                    help="Flask App Port")
    PARSER.add_argument("--debug",
                        type=bool,
                        default=False,
                        help="Flask App Debug")
    return PARSER.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)

