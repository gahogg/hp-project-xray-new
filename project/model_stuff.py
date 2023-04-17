import tensorflow as tf
import cv2
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.applications.inception_v3  import preprocess_input
from tensorflow.keras.models import load_model

CLASS_MAPPING = ['Covid', 'Normal', 'Pneumonia']

def get_heatmap_tensor(img_array, model):
    x = img_array
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    
    with tf.GradientTape() as tape:
      last_conv_layer = model.get_layer('conv2d_93')
      iterate = tf.keras.models.Model([model.inputs], [model.output, 
                                                       last_conv_layer.output])
      model_out, last_conv_layer = iterate(x)
      predicted_index = np.argmax(model_out)
      confidence = np.max(model_out)
      class_name = CLASS_MAPPING[predicted_index]
      class_out = model_out[:, np.argmax(model_out[0])]
      grads = tape.gradient(class_out, last_conv_layer)
      pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((8, 8))
    
    return heatmap, (class_name, confidence)

def get_img_with_heatmap(image_path, intensity, model):
    img_array = cv2.imread(image_path)
    heatmap, (class_name, confidence) = get_heatmap_tensor(img_array, model)
    heatmap = cv2.resize(heatmap, (299, 299))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    new_img = np.uint8((heatmap * intensity) + img_array)
    
    return new_img, (class_name, confidence)

loaded_model = load_model('model/')
