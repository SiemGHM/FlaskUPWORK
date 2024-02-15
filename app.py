from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
import io
import os
import uuid
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    # Check if the POST request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read image file
        image = Image.open(io.BytesIO(file.read()))

        # Perform image processing (e.g., convert to grayscale)
        processed_image = image.convert('L')

        # Create a directory named 'tumor_1' if it doesn't exist
        directory = os.path.join(app.root_path, 'tumor_1')
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Generate a unique filename for the processed image
        filename = str(uuid.uuid4()) + '.jpg'

        # Save the processed image to a file in the 'tumor_1' directory
        processed_image_path = os.path.join(directory, filename)
        processed_image.save(processed_image_path)

        # Convert processed image to base64 string
        with open(processed_image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

        # Return JSON response with message and processed image data
        return jsonify({'message': 'Image processed and saved successfully', 'image_data': encoded_string}), 200

@app.route('/show_processed_image/<filename>')
def show_processed_image(filename):
    # Return the processed image file
    return send_file(os.path.join(app.root_path, 'tumor_1', filename), mimetype='image/jpg')

if __name__ == '__main__':
    app.run(debug=True)
