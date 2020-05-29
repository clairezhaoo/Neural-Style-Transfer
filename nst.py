# import and configure models
import tensorflow as tf
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype = np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor  = tensor[0]
    return PIL.Image.fromarray(tensor)

# download images and choose a style and content image
content_path = tf.keras.utils.get_file('seychelles.jpg', 'file:///Users/clairezhao/Desktop/seychelles.webp')
style_path = tf.keras.utils.get_file('kandinsky.jpg','file:///Users/clairezhao/Desktop/kadinsky.jpg')

# visualize the input
def load_img(path_to_img):     # load an image and limit the max dimesion to 512 pixels
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# display an image
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if(title):
        plt.title(title)

content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1,2,1)
imshow(content_image, 'Content Image')
plt.subplot(1,2,2)       # what is plot?
imshow(style_image, 'Style Image')

#-----------------------------
import tensorflow_hub as hub
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image)


# define content and style representations
# use intermediate layers of the model to get the content and style
# first layer activiations represent features like edges and textures
# final laywer s represent higher-level features like object parts

# for the input image, match style and content representations @ the intermediate layers

# load and test the VGG19 - pretrained image classification network
x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
X = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
prediction_probabilities.shape

# i think this identifies the image, and shows the probability?
predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]

# load a VGG19 without classification and list layer names
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
print()
for layer in vgg.layers:
    print(layer.name)

# choose the intermediate layers (represent style and content)
content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv2']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# build the model ...
# define a model like -->  model = Model(inputs, outputs)
def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values """
    # load the pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

# create the model
style_extract = vgg_layers(style_layers)
style_outputs = style_extracot(style_image*255)

# analyze the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
    print(name)
    print("   shape: ", output.numpy().shape)
    print("   min: ", output.numpy().min())
    print("   max: ", output.numpy().max())
    print("   mean: ", output.numpy().mean())
    print()

# calculating the style
# content is represented by values of intermedeiate feature maps
# style can be described by the means and correlations across the different feature maps
# calculate a Gram matriz that includes all this info
# - by taking the outer product of the feature vector with iteself at each location
# - and averaging that outerproduct over all locations
# Gram matrix can be implemented using tf.linalg.einsum
def gram_matrix(input_tensor):
    result = tf.linag.einsum('bijc,bujd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

# extract style and content
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

        def call(self, inputs):
            "Expect float input in [0,1]"
            inputs = inputs*255.0
            preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
            outputs = self.vgg(preprocessed_input)
            style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                              outputs[self.num_style_layers:])
            style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

            # content_dict =
