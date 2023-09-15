import cv2

img_location = 'C:/Users/Chirag/OneDrive/Desktop/LA PROJECT/static/uploads/'
filename = 'image1.jpg'

#reads the image
img = cv2.imread(img_location + filename)

#Convert the image into gray scalr and inverted gray scale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
inverted_gray = 255 - gray_image

#Convert the gray image into blurred gray image and invert it
blurred_gray = cv2.GaussianBlur(gray_image,(21,21),0)
inverted_blur = 255 - blurred_gray

#pencil sketch
pencil_sketch = cv2.divide(gray_image, blurred_gray, scale = 256.0)

#For sketch with blue highlight
blur_img = cv2.GaussianBlur(img,(21,21),0)
sketch = cv2.divide(img, blur_img, scale = 256.0)

#For canny image 
imgCanny = cv2.Canny(img, 100, 150)

#Display the image 
cv2.imshow('Pencil sketch',pencil_sketch)
cv2.imshow('Original image',img)
cv2.imshow('gray image',gray_image)
cv2.imshow('Inverted gray',inverted_gray)
cv2.imshow('inverted blur image',inverted_blur)
cv2.imshow('blur image',blur_img)
cv2.imshow('sketch with blue filter',sketch)
cv2.imshow('Canny',imgCanny)

cv2.waitKey()
cv2.destroyAllWindows()




