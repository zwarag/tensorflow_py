import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Use OpenCV to read and resize the image file

image = cv2.imread('Gutenberg.jpg')
resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA) / 255.0

# Now use TensorFlow to manipulate the resized_image array

image_tf = tf.Variable(resized_image, dtype='float32')

image_tf_red = tf.Variable(tf.zeros(image_tf.shape), dtype='float32')
print('-----------------------------')
print(image_tf_red)

# image_tf_red = image_tf_red[:, :, 0].assign(image_tf[:, :, 0])
image_tf_red[:, :, 0].assign(image_tf[:, :, 0])
print('-----------------------------')
print(image_tf_red)
print('-----------------------------')

# Use Matplotlib to plot the images

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
plt.suptitle('Images')

axs[0].imshow(resized_image)
axs[0].set_title('Resized image', fontsize=10)
axs[0].set_xlabel('x pixel', fontsize=10)
axs[0].set_ylabel('y pixel', fontsize=10)

axs[1].imshow(image_tf)
axs[1].set_title('TensorFlow variable image', fontsize=10)
axs[1].set_xlabel('x pixel', fontsize=10)
axs[1].set_ylabel('y pixel', fontsize=10)

axs[2].imshow(image_tf_red)
axs[2].set_title('TensorFlow variable red image', fontsize=10)
axs[2].set_xlabel('x pixel', fontsize=10)
axs[2].set_ylabel('y pixel', fontsize=10)

plt.show()
