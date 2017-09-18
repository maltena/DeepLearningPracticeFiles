
# coding: utf-8

# ## Convolutional Filter Visualization
# 
# Based on fchollet
# https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
# 
# 
# Visualization of the filters of VGG16, via gradient ascent in input space.
# 
# Results example: http://i.imgur.com/4nj4KjN.jpg
# Weights file:  https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
#
# -----------------------------------------------------------------------------------------------
# RUNNING INSTRUCTIONS
# Set the layer to be probed and the probe image dimensions below. Save and then:
#  in terminal, run this with the following command:
#      python ./Conv_Filter_Visualization-Exported.py
#	python ./Conv_Filter_Vis.py
# ----------------------------------------------------------------------------------------------

# In[1]:

# Initialization

from __future__ import print_function

from scipy.misc import imsave  # Save an array as an image.  scipy.misc.imsave(name, arr, format=None)
import numpy as np
import time
import math as m


# In[2]:

from keras.applications import vgg16
from keras import backend as K   # If you want the Keras modules you write to be compatible with both 
# Theano (th) and TensorFlow (tf), you have to write them via the abstract Keras backend API.


# In[3]:

# util function to convert a tensor into a valid image

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)  #Assume the 1e-5 is a really small value to make sure there never a divide by 0 condition
    x *= 0.1
    
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    
    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1,2,0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# In[4]:

# build the VGG network with Imagenet weights
model = vgg16.VGG16(weights='imagenet', include_top=False)
print ('VGG16 Model Loaded.')

model.summary()  # outputs a nice, compact, readable summary of the model!!!! :D


# --------------------------------------------------------------------------------------------------------------
# In[5]:  SET THE MODEL LAYER HERE BEFORE RUNNING ------------------------------------------------------------

# Specify the layer we want to visualize
model_name = 'VGG16'
# (see model definiation at keras/applications/vgg16.py
# also see technical paper at https://arxiv.org/pdf/1409.1556.pdf
layer_name = 'block5_pool'

# look at the list above and enter a number not exceeding  the rightmost value of the shape
filters_in_layer  = 300  # recommend this not exceed 200


# In[6]:  SET THE DIMENSIONS OF THE PROBE IMAGE HERE BEFORE RUNNING -------------------------------------------

# dimensions of the generated picture for each filter
#img_width = 128
#img_height = 128
#img_width, img_height  = 128, 128  # just trying this out to verify this also works
img_width, img_height  = 128, 128

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

# In[7]:

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names)
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
# print (layer_dict)


# In[8]:

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5) #Assume 1e-5 is tiny value to ensure no divide by 0 condition


# In[9]:

kept_filters = [] # define an empty list in python

for filter_index in range(0, filters_in_layer):
    # we scan only through the first 200 filters
    # but there are actually 512 of them
    # from the model summary:  block5_conv1 (Conv2D)        (None, None, None, 512)   2359808 
    print ('Processing filter %d' % filter_index)
    start_time = time.time()
    
    # we build a loss funciton that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    
    if K.image_data_format() == 'channels_first':   # appears to be a format check.need to understand where needed
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])
        
    # we compute the gradient of the input picture wrt to this loss
    grads = K.gradients(loss, input_img)[0]
    
    # normalization trick - normalize the gradient
    grads = normalize(grads)
    
    # this function returns the loss and grad given the input picture
    iterate = K.function([input_img], [loss, grads])
    
    # step size for the gradient ascent (AScent!)
    step= 1
    
    # we start with a 'gray' image with some random noise  (again, some kind of dimensionality check below)
    if K.image_data_format() == 'channels_first':       
        input_img_data = np.random.random((1, 3, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128
    # random input to image data has range (0 to 1). 
    # This recenters (-.5 to +.5), *20 = (-10 to +10), then +128 yields gray img with range (118 to 138)
    
    # run gradient ascent for 20 steps - NOTE:  Update to 40+ steps for later layers
    ascent_loop = 60
    for i in range (ascent_loop):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
        print ('Current loss value: ', loss_value)
        
        if loss_value < 0.0:
            # some filters get stuck to 0, so skip these
            break
        if i > 0 and i < 5 and loss_value < .02:   # added another check, because usually if it starts at 0, it stays there
            break  # this works well for i>0 for early layers.  May want to increase for later layers? Probably no
		# added a second i<5 because sometimes the loss_value decreases, but maybe it starts climbing again?
            
    # decode the resulting input image
    # decode the resulting input image
    if loss_value > 0.0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print ('Filter %d processed in %ds' % (filter_index, end_time - start_time))


# In[10]:

n = 8  # it seems there aren't enough pics that converge if you go to the range of filters elicited

# The number of filters in the layer has implications as to how we should display them.
# Program is set up to make an (n x n) square of the filters, therefore:
n = m.trunc (m.sqrt(len(kept_filters)))
print ('number of converged images (e..g, filter excitations is: ', len(kept_filters))
print ('output should be a square of dimension: ', n, 'x', n)


# In[12]:

# stitch the best filters into an array for display

# the filters that have the highest loss are assumed to be better looking
# THIS MIGHT BE A PLACE TO EXPERIMENT???
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for the 8x8 filters of size 128x128, with a 5px margin between filters
margin = 5
displayed_width = n * img_width + (n-1) * margin
displayed_height = n * img_height + (n-1) * margin
displayed_filters = np.zeros((displayed_width, displayed_height, 3))

# fill in the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        displayed_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
        
# save the result to disk
saved_image_name = ('filters_' + model_name + '_' + layer_name + '_probed_by_' 
                    + str(img_height) + 'x' + str(img_width) + '_loop_' + str(ascent_loop) +'.png')

imsave (saved_image_name, displayed_filters)

print ('File saved to: ', saved_image_name)


# In[ ]:




# In[ ]:



