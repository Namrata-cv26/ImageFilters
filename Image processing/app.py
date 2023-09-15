import os
import io
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_file
import cv2

app = Flask(__name__)

def svd_filter(img):
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Perform SVD decomposition
    U, S, V = np.linalg.svd(img_array, full_matrices=False)
    
    # Set a percentage of singular values to keep
    keep_percent = 0.1
    k = int(keep_percent * len(S))
    
    # Reconstruct the image using the k largest singular values
    reduced_img_array = np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k, :]))
    reduced_img_array[reduced_img_array < 0] = 0
    reduced_img_array[reduced_img_array > 255] = 255
    
    # Convert back to PIL image
    reduced_img = Image.fromarray(reduced_img_array.astype('uint8'), 'RGB')
    
    return reduced_img

def canny_edge_detection(img):
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    
    # Detect edges using Canny Edge Detection
    edges = cv2.Canny(blur, 100, 200)
    
    # Convert back to PIL image
    edges_img = Image.fromarray(edges.astype('uint8'), 'L')
    
    return edges_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    filter_type = request.form['filter_type']
    file = request.files['file']
    img = Image.open(file)
    if filter_type == 'grayscale':
        img = img.convert('L')
    elif filter_type == 'negative':
        img = ImageOps.invert(img)
    elif filter_type == 'sepia':
        img = ImageOps.colorize(img.convert('L'), '#704214', '#C1B282')
    elif filter_type == 'brightness':
        if 'factor' in request.form:
            factor = request.form['factor']
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(float(factor))
    elif filter_type == 'contrast':
        if 'factor' in request.form:
            factor = request.form['factor']
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(float(factor))
    elif filter_type == 'sharpness':
        if 'factor' in request.form:
            factor = request.form['factor']
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(float(factor))
    elif filter_type == 'svd':
        img = svd_filter(img)
    elif filter_type == 'gaussian_blur':
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
    elif filter_type == 'canny_edge':
        img = canny_edge_detection(img)
    elif filter_type == 'invert':
        img = ImageOps.invert(img)
    elif filter_type == 'color_balance':
       if 'factor' in request.form:
        factor = request.form['factor']
        img = ImageOps.colorize(img.convert('L'), (128, 128, 128), (int(factor), int(factor), int(factor)))
    elif filter_type == 'saturation':
     if 'factor' in request.form:
        factor = request.form['factor']
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(float(factor))
    elif filter_type == 'hue':
     if 'factor' in request.form:
        factor = request.form['factor']
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.0)
        img = ImageOps.colorize(img.convert('L'), (int(factor), 0, 0), (255, int(factor), 255))
    elif filter_type == 'emboss':
       img = img.filter(ImageFilter.EMBOSS)

    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
