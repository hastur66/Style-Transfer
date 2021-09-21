import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)

	file = request.files['file']
	
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
		
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/Style' + filename), code=301)

@app.route('/upload2', methods=['POST'])
def upload_image2():
	if 'file2' not in request.files:
		flash('No file part')
		return redirect(request.url)

	file2 = request.files['file2']
	
	if file2.filename2 == '':
		flash('No image selected for uploading')
		return redirect(request.url)
		
	if file2 and allowed_file(file2.filename2):
		filename2 = secure_filename(file2.filename2)
		file2.save(os.path.join(app.config['UPLOAD_FOLDER2'], filename2))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('upload2.html', filename=filename2)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)


@app.route('/display/<filename>')
def display_image2(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/Content' + filename), code=301)


@app.route('/artwork', methods=['GET', 'POST'])
def art_work():
	#if request.method == 'GET':
	#	return redirect(url_for('artwork.html'))
	return render_template('artwork.html')

@app.route('/upload2', methods=['GET', 'POST'])
def upload2():
	#if request.method == 'GET':
	#	return redirect(url_for('artwork.html'))
	return render_template('upload2.html')

from style_transfer_tf_hub import *

tensor_img = tensor_to_image()

load_path()

load_img()

import tensorflow_hub as hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

model = tf.keras.Sequential(hub_model)
tf.saved_model.save(model, './model')

stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image)

if __name__ == "__main__":
    app.run()