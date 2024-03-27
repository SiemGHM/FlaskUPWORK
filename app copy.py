from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
import io
import os
import uuid
import base64
import matlab.engine

import tempfile




eng = matlab.engine.start_matlab()
eng.addpath('/Users/siemghirmay/Documents/Upwork/Job-1', nargout=0)

imageDir = '/Users/siemghirmay/Documents/Upwork/Job-1/'
processedDir = '/Users/siemghirmay/Documents/Upwork/Job-1/processed/'
apiUrl = 'https://flask-conversion-9bc055773d8c.herokuapp.com/process_image'
# processAndPostImages(imageDir, processedDir, apiUrl);

# temp_file_path = '/Users/siemghirmay/Documents/Upwork/Job-1'
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Create a temporary directory to save the uploaded image
    with tempfile.TemporaryDirectory() as tempdirO:
        tempdir = tempfile.mkdtemp()
        filenamee = str(uuid.uuid4()) + '.jpeg'
        image_path = os.path.join(tempdir, filenamee)
        file.save(image_path)

        # Directory where processed images will be saved
        processed_dir = tempfile.mkdtemp()

        # Assuming `apiUrl` is defined elsewhere in your Flask app
        apiUrl = 'http://your-flask-api-endpoint/processed_image'
        
        # Call the MATLAB function
        print(tempdir, processed_dir, apiUrl)
        eng.processAndPostImages(str(tempdir), str(processed_dir), apiUrl, nargout=0)

        # Assuming your MATLAB function saves processed images in a specific manner,
        # you would then read the processed image from `processed_dir` and return it

        # For demonstration, let's assume there's only one processed image
        processed_images = os.listdir(processed_dir)
        print(processed_images)
        image = Image.open(processed_dir + '/' + processed_images[0])
        image.save('processed/'+ file.filename.split('.')[0] + filenamee)
        
        # Convert processed image to base64 string
        with open('processed/'+ file.filename.split('.')[0] + filenamee, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

        # Return JSON response with message and processed image data
        return jsonify({'message': 'Image processed and saved successfully', 'image_data': encoded_string}), 200
        # if processed_images:
        #     return send_file('processed/'+ file.filename.split('.')[0] + filenamee)
        #     # return send_file(os.path.join(processed_dir, processed_images[0]), mimetype='image/jpeg')
        # else:
        #     return jsonify({'error': 'Processing failed or no images processed'}), 500

# Save the processed image to a file in the 'tumor_1' directory
#         processed_image_path = os.path.join(directory, filename)
#         processed_image.save(processed_image_path)

#         # Convert processed image to base64 string
#         with open(processed_image_path, "rb") as img_file:
#             encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

#         # Return JSON response with message and processed image data
#         return jsonify({'message': 'Image processed and saved successfully', 'image_data': encoded_string}), 200






# @app.route('/process_image', methods=['POST'])
# def process_image():
#     # Check if the POST request has the file part
#     if 'image' not in request.files:
#         return jsonify({'error': 'No file part in the request'}), 400

#     file = request.files['image']

#     # If the user does not select a file, the browser submits an empty file without a filename
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     if file:
#         # Read image file
#         image = Image.open(io.BytesIO(file.read()))

#         # Perform image processing (e.g., convert to grayscale)
#         processed_image = image.convert('L')
#         eng.processAndPostImages('/Users/siemghirmay/Documents/Upwork/Job-1/', processedDir, apiUrl, nargout=0)
#         # print(result)
#         # eng.eval("run('/Users/siemghirmay/Documents/Upwork/Job-1/img_processing.m')", nargout=0)

#         # Create a directory named 'tumor_1' if it doesn't exist
#         directory = os.path.join(app.root_path, 'tumor_1')
#         if not os.path.exists(directory):
#             os.makedirs(directory)

#         # Generate a unique filename for the processed image
#         filename = str(uuid.uuid4()) + '.jpg'

#         # Save the processed image to a file in the 'tumor_1' directory
#         processed_image_path = os.path.join(directory, filename)
#         processed_image.save(processed_image_path)

#         # Convert processed image to base64 string
#         with open(processed_image_path, "rb") as img_file:
#             encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

#         # Return JSON response with message and processed image data
#         return jsonify({'message': 'Image processed and saved successfully', 'image_data': encoded_string}), 200

@app.route('/show_processed_image/<filename>')
def show_processed_image(filename):
    # Return the processed image file
    return send_file(os.path.join(app.root_path, 'tumor_1', filename), mimetype='image/jpg')

if __name__ == '__main__':
    app.run(debug=True)
