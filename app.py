from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
import io
import os
import uuid
import base64
from oct2py import Oct2Py
oc = Oct2Py()

# Assuming your Octave function is saved in 'processImages.m'
current_directory = os.path.dirname(os.path.realpath(__file__))

oc = Oct2Py()
# Add the directory to Octave's path
oc.addpath(current_directory)


import tempfile


import os
import cv2
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import savetxt
from tensorflow import keras
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

import json







labels = ['no_tumor', 'tumor']
class_map = {
    'no_tumor': 0,
    'tumor': 1,
    'no_tumor_test': 0,
    'tumor_test': 1
}

train_img = []
train_labels = []

test_img = []
test_labels = []
path_train = 'noncancer7/train'
path_test = 'noncancer7/test'

img_size = 128

for i in os.listdir(path_train):
    for j in os.listdir(path_train+'/'+i):
        img_raw = cv2.imread(path_train+'/'+i+'/'+j)
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        resize_img = cv2.resize(img, (img_size, img_size))
        train_img.append(resize_img)
        train_labels.append(class_map[i])

for i in os.listdir(path_test):
    for j in os.listdir(path_test+'/'+i):
        img_raw = cv2.imread(path_test+'/'+i+'/'+j)
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        resize_img = cv2.resize(img, (img_size, img_size))
        test_img.append(resize_img)
        test_labels.append(class_map[i])

train_img = np.array(train_img)
train_labels = np.array(train_labels)
test_img = np.array(test_img)
test_labels = np.array(test_labels)

# # %%
print("Shape of full train set is:", train_img.shape)
print("Shape of full test set is:", test_img.shape)

# %%
X_train = train_img
y_train = train_labels
X_test = test_img
y_test = test_labels

# %%
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
model = Sequential()

model.add(Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2,2), padding="valid"))

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2,2), padding="valid"))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
checkpoint = ModelCheckpoint("tumor_research.keras", monitor="accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)



# hist = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1,
#                 callbacks=[checkpoint,reduce_lr])

# # %%
# y_pred = model.predict(X_test)

# # %%
# y_predtrain = model.predict(X_train)

# # %%


# y_test = np.argmax(y_test,axis=1)
# y_pred = np.argmax(y_pred, axis=1)

# import itertools

# target_names = ['no tumor', 'tumor']



# def plot_confusion_matrix(cm, classes,
#                           normalize = False,
#                           title = 'Confusion matrix',
#                           cmap = plt.cm.Blues):

#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting 'normalize=True'.
#     """

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')


# cnf_matrix = (confusion_matrix(y_test, y_pred))

# # np.set_printoptions(precision=3)

# # plt.figure()

# # plot_confusion_matrix(cnf_matrix, classes=target_names,
# #                       title='Confusion matrix')

# # plt.show()
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# # %%
# plt.plot(fpr,tpr)
# plt.title('ROC Curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.show()

# # %%
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve



# # Calculate the ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# # Convert predicted probabilities to binary predictions (0 or 1)
# # You can choose a threshold to determine the binary classification based on your problem
# # For example, you can use a threshold of 0.5, but adjust it according to your specific problem
# threshold = 0.5
# y_pred_binary = [1 if pred >= threshold else 0 for pred in y_pred]

# # Calculate accuracy, precision, recall, and F1-score
# accuracy = accuracy_score(y_test, y_pred_binary)
# precision = precision_score(y_test, y_pred_binary)
# recall = recall_score(y_test, y_pred_binary)
# f1 = f1_score(y_test, y_pred_binary)

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)


 

#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# eng = matlab.engine.start_matlab()
# eng.addpath('/Users/siemghirmay/Documents/Upwork/Job-1', nargout=0)

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
    # eng = matlab.engine.start_matlab()
    
    
    filenamee = str(uuid.uuid4()) + '.jpeg'
    testsave = os.path.join('tumor_1/Test', file.filename.split('.')[0] +"_original_"+ filenamee)
    trainsave = os.path.join('tumor_1/Train', file.filename.split('.')[0][0:4] +"_original_"+ filenamee)
    file.save(testsave)
    file.seek(0)
    file.save(trainsave)
    file.seek(0)
    print("File saved")
    # Create a temporary directory to save the uploaded image
    with tempfile.TemporaryDirectory() as tempdirO:
        tempdir = tempfile.mkdtemp()
        
        image_path = os.path.join(tempdir, filenamee)
        file.save(image_path)
        file.seek(0)


        # Directory where processed images will be saved
        processed_dir = tempfile.mkdtemp()

        # Assuming `apiUrl` is defined elsewhere in your Flask app
        apiUrl = 'http://your-flask-api-endpoint/processed_image'
        
        # Call the MATLAB function
        print(tempdir, processed_dir, apiUrl)
        ress = oc.processImages(str(tempdir), str(processed_dir))

        # Assuming your MATLAB function saves processed images in a specific manner,
        # you would then read the processed image from `processed_dir` and return it

        # For demonstration, let's assume there's only one processed image
        processed_images = os.listdir(processed_dir)
        print(processed_images)
        image = Image.open(processed_dir + '/' + processed_images[0])
        image.save('processed/'+ file.filename.split('.')[0] + filenamee)
        image.seek(0)
        image.save('tumor_1/Test/Processed/'+ file.filename.split('.')[0] + filenamee)
        image.seek(0)
        image.save('tumor_1/Train/Processed/'+ file.filename.split('.')[0] + filenamee)
        image.seek(0)
        
        
        # Convert processed image to base64 string
        with open('processed/'+ file.filename.split('.')[0] + filenamee, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

        # Return JSON response with message and processed image data
        # return jsonify({'message': 'Image processed and saved successfully', 'image_data': encoded_string}), 200
        
    
    
    
        train_img = []
        train_labels = []

        test_img = []
        test_labels = []

        path_train = 'noncancer7/train'
        path_test = 'noncancer7/test'

        img_size = 128

        for i in os.listdir(path_train):
            for j in os.listdir(path_train+'/'+i):
                img_raw = cv2.imread(path_train+'/'+i+'/'+j)
                img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                # img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                resize_img = cv2.resize(img, (img_size, img_size))
                train_img.append(resize_img)
                train_labels.append(class_map[i])

        for i in os.listdir(path_test):
            for j in os.listdir(path_test+'/'+i):
                img_raw = cv2.imread(path_test+'/'+i+'/'+j)
                img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                # img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                resize_img = cv2.resize(img, (img_size, img_size))
                test_img.append(resize_img)
                test_labels.append(class_map[i])

        train_img = np.array(train_img)
        train_labels = np.array(train_labels)
        test_img = np.array(test_img)
        test_labels = np.array(test_labels)

        # %%
        print("Shape of full train set is:", train_img.shape)
        print("Shape of full test set is:", test_img.shape)

        # %%
        X_train = train_img
        y_train = train_labels
        X_test = test_img
        y_test = test_labels

        # %%
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        # %%
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        hist = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1,
                    callbacks=[checkpoint,reduce_lr])

        # %%
        y_pred = model.predict(X_test)

        # %%
        y_predtrain = model.predict(X_train)
        
        y_test = np.argmax(y_test,axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        cnf_matrix = (confusion_matrix(y_test, y_pred))
        # cnf_matrix = np.array([[4, 3], [3, 5]])
    
        # Convert the confusion matrix to a format easy to use in HTML/JS
        cnf_matrix_list = cnf_matrix.tolist() 
        return jsonify({'message': 'Image processed and saved successfully', 'image_data': encoded_string, 'confusion_matrix':cnf_matrix_list}), 200
        return render_template('cnf.html', confusion_matrix=cnf_matrix_list)
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
# @app.route('/start')
# def start():
#     return render_template('upload.html')


# @app.route('/trainModel', methods=['POST'])
# def trainModel():
#     # hist = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1,
#     #             callbacks=[checkpoint,reduce_lr])

#     # # %%
#     # y_pred = model.predict(X_test)

#     # # %%
#     # y_predtrain = model.predict(X_train)
    
#     # y_test = np.argmax(y_test,axis=1)
#     # y_pred = np.argmax(y_pred, axis=1)
#     # cnf_matrix = (confusion_matrix(y_test, y_pred))
    
#     cnf_matrix = np.array([[5, 2], [3, 5]])
    
#     # Convert the confusion matrix to a format easy to use in HTML/JS
#     cnf_matrix_list = cnf_matrix.tolist() 
#     return render_template('cnf.html', confusion_matrix=cnf_matrix_list)




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
#             encoded_string = base64.b64encode(img_file.read()).decode('utf-8') apshaiderbukhari786@gmail.com

#         # Return JSON response with message and processed image data
#         return jsonify({'message': 'Image processed and saved successfully', 'image_data': encoded_string}), 200

@app.route('/show_processed_image/<filename>')
def show_processed_image(filename):
    # Return the processed image file
    return send_file(os.path.join(app.root_path, 'tumor_1', filename), mimetype='image/jpg')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if $PORT not set
    app.run(host='0.0.0.0', port=port)
