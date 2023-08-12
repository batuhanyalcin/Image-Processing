#!/usr/bin/env python
# coding: utf-8

# # Assignment 1: Camera Pipeline
# ## Batuhan Yalçın 64274
# ### March 29, 2023

# ## Necessary imports

# In[32]:


import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sc
import skimage as sk
from skimage import color
import plotly.express as px
import copy


# ## Reading Image

# In[33]:


imgr = io.imread('campus.tiff')

##properties of the image
print("Bits per pixel:", imgr.dtype.itemsize*8)
print("Image width dimension:", imgr.shape[1])
print("Image height dimension:",  imgr.shape[0])
print("Data type:" , imgr.dtype)

##Then converting double 
img_double = imgr.astype(np.double)
print(img_double)


# ## Linearization

# In[34]:


black = 150
white = 4095

# linear transformation 
#Shift
linear_img_shift = (img_double - black) 
#Scale 
linear_img = linear_img_shift  / (white - black)

print(linear_img)
# Clip negative values to 0 and values greater than 1 to 1
linear_img = np.clip(linear_img, 0, 1)
print(linear_img)


# ## Identifying the correct Bayer pattern

# In[35]:


### First try to find correct pattern using top square 

# Extract the top-left 2x2 square of the image
square = linear_img[:2, :2]
print(square)
plt.imshow(square)

# it seems purple green yellow and again purple but how this can be possible answer micture of the colors
#Purple can be mixture of the blue and red
#Green can be micture of the blue and yellow
#yellow can be mixture of the red and green 
# The reflection comes from the object can effect these colors so it can be multiple bayer pattern but since 1. and 4. points
#should consist blue and red and it only in RGGB pattern it can more suitable lets do further analysis.



# #### investigating the only red areas.

# In[36]:


fig = px.imshow(img_double)
fig.show()


# In[37]:


# Extract the top-left 2x2 square of the image in red area
square = linear_img[896:898, 2070:2072]
print(square)
plt.imshow(square)

#when you remove the red combinations from all colors it gives, almost red, green, green, blue.


# In[38]:


# Since it written in pdf after white balancing its more easy to find correct bayern pattern I use an automatic grayscale assumption
#Note this technique not same with the given in lecture lecture method will used in the next parts
# References White Balancing — An Enhancement Technique in Image Processing/ Matt Mulian
img_wb = linear_img / np.mean(linear_img, axis=(0,1))

red = img_wb[::2, ::2]
green1 = img_wb[::2, 1::2]
green2 = img_wb[1::2, ::2]
blue =  img_wb[1::2, 1::2]
green = (green1+green2)/2

plt.imshow(square)


image_test=np.dstack((red,green,blue))
plt.imshow(image_test) # draw first image

## Congrulations!!!! it gives red are as same as red area


# In[40]:


## additional approach found at the ethernet and seems the logical 
## It basicly calculates means values of all channels and comapres with the expected values of the original channels then 
## which bayer patterns shows the smallest error can give the correct result

channel1 = square[::2, ::2]
channel2 = square[::2, 1::2]
channel3 = square[1::2, ::2]
channel4 =  square[1::2, 1::2]

# Calculate the mean values of the red, green, and blue channels in the square
mean_first = np.mean(channel1)
mean_seconds = np.mean((channel2, channel3))
mean_third = np.mean(channel4)


#Take differences between expected value 
RGGB = np.abs([mean_first-1, mean_seconds-0.5, mean_third-0.5])
GBGR = np.abs([mean_first-0.5, mean_seconds-1, mean_third-0.5])
BGRG = np.abs([mean_first-0.5, mean_seconds-0.5, mean_third-1])
GRBG = np.abs([mean_first-0.5, mean_seconds-1, mean_third-1])

#Sum of errors
e_RGGB = np.sum(RGGB)
e_GBGR = np.sum(GBGR)
e_BGRG = np.sum(BGRG)
e_GRBG = np.sum(GRBG)
errors = [e_RGGB,e_GBGR,e_BGRG,e_GRBG]
print(errors)

#Finding best bayer pattern
bayers = ['RGGB', 'GBGR', 'BGRG', 'GRBG']
bayer = bayers[np.argmin(errors)]
print('Bayyern Pattern is following due to most less error:')
print(bayer)


# ### Final proof

# In[41]:


# Since it written in pdf after white balancing its more easy to find correct bayern pattern I use an automatic grayscale assumption
#Note this technique not same with the given in lecture lecture method will used in the next parts
# References White Balancing — An Enhancement Technique in Image Processing/ Matt Mulian
img_wb = linear_img / np.mean(linear_img, axis=(0,1))

red = img_wb[::2, ::2]
green1 = img_wb[::2, 1::2]
green2 = img_wb[1::2, ::2]
blue =  img_wb[1::2, 1::2]
green = (green1+green2)/2
image_rggb=np.dstack((red,green,blue))


green1 = img_wb[::2, ::2]
blue = img_wb[::2, 1::2]
green2 = img_wb[1::2, ::2]
red =  img_wb[1::2, 1::2]
green = (green1+green2)/2
image_gbgr=np.dstack((red,green,blue))


blue = img_wb[::2, ::2]
green1 = img_wb[::2, 1::2]
red = img_wb[1::2, ::2]
green2 =  img_wb[1::2, 1::2]
green = (green1+green2)/2
image_bgrg = np.dstack((red,green,blue))

green1 = img_wb[::2, ::2]
red = img_wb[::2, 1::2]
blue = img_wb[1::2, ::2]
green2 =  img_wb[1::2, 1::2]
green = (green1+green2)/2
image_grbg=np.dstack((red,green,blue))
plt.imshow(image_test) # draw first image


# display all images in a 2x2 grid
fig = plt.figure() # create a new figure
fig.add_subplot(2, 2, 1) # draw RGGB
plt.imshow(image_rggb) 
fig.add_subplot(2, 2, 2) # draw GBGR
plt.imshow(image_gbgr)
fig.add_subplot(2, 2, 3) # draw BGRG
plt.imshow(image_bgrg)
fig.add_subplot(2, 2, 4) # draw GRBG
plt.imshow(image_grbg)
plt.savefig('correct_bayer.png') # saves current figure as a PNG file
plt.show() # displays figure

## Congrulations!!!! only RGGB gives red are as same as red area


# ### Correct bayer pattern

# In[42]:


img_byr=linear_img
red = img_byr[::2, ::2]
green1 = img_byr[::2, 1::2]
green2 = img_byr[1::2, ::2]
blue =  img_byr[1::2, 1::2]
green = (green1+green2)/2
image_rgb_after_bayer =np.dstack((red,green,blue))
plt.imshow(image_rgb_after_bayer)



# ## White balancing

# In[43]:


img_rgb = image_rgb_after_bayer

#Avarage and Max Values for algorithms
r_avg = np.mean(img_rgb[:, :, 0])
r_max = np.max(img_rgb[:, :, 0])

g_avg = np.mean(img_rgb[:, :, 1])
g_max = np.max(img_rgb[:, :, 1])

b_avg = np.mean(img_rgb[:, :, 2])
b_max = np.max(img_rgb[:, :, 2])




# Gray world
grywrld_matrix = np.array([[(g_avg / r_avg), 0, 0],
                             [0, 1, 0],
                             [0, 0, (g_avg / b_avg)]])
img_after_gray = np.dot(img_rgb, grywrld_matrix)

# White world
whtwrld_matrix = np.array([[(g_max / r_max), 0, 0],
             [0, 1, 0],
             [0, 0, (g_max / b_max)]])

img_after_white = np.matmul(img_rgb, whtwrld_matrix)


# Preset scaling
r_scale = 2.393118
g_scale = 1.0
b_scale = 1.223981
img_after_pre = img_rgb
img_after_pre[:, :, 0] = img_after_pre[:, :, 0] * r_scale
img_after_pre[:, :, 1] = img_after_pre[:, :, 1] * g_scale
img_after_pre[:, :, 2] = img_after_pre[:, :, 2] * b_scale


# In[44]:


# display all images in a 1x3 grid
fig = plt.figure() # create a new figure
fig.add_subplot(1, 3, 1)
plt.imshow(img_after_white) # draw first image
fig.add_subplot(1, 3, 2) # draw second image
plt.imshow(img_after_gray)
fig.add_subplot(1, 3, 3) # draw third image
plt.imshow(img_after_pre)
plt.savefig('white_balance.png') # saves current figure as a PNG file
plt.show() # displays figure


# In[45]:


chsn_img = img_after_gray


# ## Demosaicing
# 

# In[46]:


height = np.shape(chsn_img)[0]
width = np.shape(chsn_img)[1]

print(width)
dmsc_r_func = sc.interp2d(np.array(range(width)), np.array(range(height)), chsn_img[:, :, 0], kind='linear')
dmsc_g_func = sc.interp2d(np.array(range(width)), np.array(range(height)), chsn_img[:, :, 1], kind='linear')
dmsc_b_func = sc.interp2d(np.array(range(width)), np.array(range(height)), chsn_img[:, :, 2], kind='linear')

dmsc_img_r = dmsc_r_func(np.arange(0, width, 0.5), np.arange(0, height, 0.5))
dmsc_img_g = dmsc_g_func(np.arange(0, width, 0.5), np.arange(0, height, 0.5))
dmsc_img_b = dmsc_b_func(np.arange(0, width, 0.5), np.arange(0, height, 0.5))

dmsc_img = np.dstack((dmsc_img_r, dmsc_img_g, dmsc_img_b))
print(dmsc_img.shape[1])


# In[47]:


plt.imshow(dmsc_img)
plt.savefig('demosaic.png') # saves current figure as a PNG file


# ## Color Space Corrections

# In[48]:


#adobe_cuff defined cam values for Nikon D3400 DSLR values fixed with dividing 10000
adobe_cuff_xyz = np.array([[6988,-1384,-714],
                    [-5631,13410,2447],
                    [-1485,2204,7318]])/10000

#Given RGB standarts for color spaces 
sRGB_std =np.array( [[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])

# Normalize rows to sum to 1 as described in the pdf
sRGB_to_cam = np.matmul(adobe_cuff_xyz, sRGB_std) 

#sRGB_to_cam = normalize(sRGB_to_cam, axis=1, norm='l1')
rgb_sum = np.sum(sRGB_to_cam, axis=1)

sRGB_to_cam[0, :] = sRGB_to_cam[0, :] / rgb_sum[0]    
sRGB_to_cam[1, :] = sRGB_to_cam[1, :] / rgb_sum[1]
sRGB_to_cam[2, :] = sRGB_to_cam[2, :] / rgb_sum[2] 

#Creating final image matrix multiplication the previous demosaicing with the sRGBtoCam matrix inverse transpose 
img_csc = np.matmul(dmsc_img,((np.linalg.inv(sRGB_to_cam)).T))
plt.savefig('color_space.png')
plt.imshow(img_csc)


# ## Brightness Adjustment and Gamma encoding

# In[49]:


test=copy.deepcopy(img_csc)


# In[50]:


### First Gray Scale image for brightness as like in games
scl = 2.75
gryscl_img = sk.color.rgb2gray(scl * test)
print(np.mean(gryscl_img))
plt.imshow(gryscl_img)


# In[51]:


### Then final result
test = np.clip((scl * test), 0, 1)
s_e_values = test <= 0.0031308
bigger_values = test > 0.0031308

img_result =( s_e_values*test*12.92
             + np.power((bigger_values*test),1/2.4) * (1 + 0.055)- bigger_values*0.055)
plt.imshow(img_result)
plt.show()


# In[52]:


sk.io.imsave('result.png', img_result)


# ## Compression

# In[ ]:


#Please dont run this when this run the size of the file increasing significantly
for i in range(1,100):
    string= 'result'+ str(i)+ '.jpeg'
    sk.io.imsave(string, img_result, quality=i)


# # Problem 3

# In[31]:


import skimage.exposure as exposure
import cv2
# Load the images A and B
imgA = io.imread('imageA.jpg')
imgB = io.imread('imageB.jpg')

# Convert the images to grayscale
grayA = sk.color.rgb2gray(imgA)
grayB = sk.color.rgb2gray(imgB)

# Subtract the images to obtain the difference image
diff = cv2.absdiff(grayA, grayB)
diff_original = cv2.absdiff(imgA, imgB)

# Auto adjust contrast and tone
img_gray = exposure.adjust_gamma(diff, gamma=1.0)
img_gray = exposure.equalize_adapthist(img_gray, clip_limit=0.03)
img_original = exposure.adjust_gamma(diff_original, gamma=1.0)
img_original = exposure.equalize_adapthist(img_original, clip_limit=0.03)

# Convert to RGB
rgb_img_gray = color.gray2rgb(img)

#Display the figures
fig = plt.figure() # create a new figure
fig.add_subplot(1, 3, 1)
plt.imshow(img_gray)
plt.axis('off')
fig.add_subplot(1, 3, 2)
plt.imshow(rgb_img_gray)
plt.axis('off')
fig.add_subplot(1, 3, 3)
plt.imshow(img_original)
plt.axis('off')
plt.savefig('problem3.png') # saves current figure as a PNG file
plt.show() # displays figure


# In[67]:


create_under_differen_white_balance(img_rgb,'g')


# In[66]:


create_under_differen_white_balance(img_rgb,'w')


# In[68]:


create_under_differen_white_balance(img_rgb,'p')


# In[71]:


create_under_birghtness(img_rgb,'brighter',20)


# In[72]:


#different gray  pattern
img_byr=linear_img
green1= img_byr[::2, ::2]
red = img_byr[::2, 1::2]
blue = img_byr[1::2, ::2]
green2 =  img_byr[1::2, 1::2]
green = (green1+green2)/2
image_rgb_after_bayer =np.dstack((red,green,blue))

create_under_differen_white_balance(image_rgb_after_bayer,'g')


# In[65]:


def create_under_differen_white_balance(rgb_picture,typ):
    img_rgb = image_rgb_after_bayer

    #Avarage and Max Values for algorithms
    r_avg = np.mean(img_rgb[:, :, 0])
    r_max = np.max(img_rgb[:, :, 0])

    g_avg = np.mean(img_rgb[:, :, 1])
    g_max = np.max(img_rgb[:, :, 1])

    b_avg = np.mean(img_rgb[:, :, 2])
    b_max = np.max(img_rgb[:, :, 2])
    
    if (typ=='g'):
        grywrld_matrix = np.array([[(g_avg / r_avg), 0, 0],
                             [0, 1, 0],
                             [0, 0, (g_avg / b_avg)]])
        image=np.dot(img_rgb, grywrld_matrix)
        print('gray_balance')
    elif (typ=='w'):
        whtwrld_matrix = np.array([[(g_max / r_max), 0, 0],
             [0, 1, 0],
             [0, 0, (g_max / b_max)]])
        image=np.matmul(img_rgb, whtwrld_matrix)
        print('white_balance')
    else:
        r_scale = 2.393118
        g_scale = 1.0
        b_scale = 1.223981
        img_after_pre = img_rgb
        img_after_pre[:, :, 0] = img_after_pre[:, :, 0] * r_scale
        img_after_pre[:, :, 1] = img_after_pre[:, :, 1] * g_scale
        img_after_pre[:, :, 2] = img_after_pre[:, :, 2] * b_scale
        image=img_after_pre
        print('preset_balance')
    
    plt.imshow(image)
    chnn_img=image
    height = np.shape(chsn_img)[0]
    width = np.shape(chsn_img)[1]


    dmsc_r_func = sc.interp2d(np.array(range(width)), np.array(range(height)), chsn_img[:, :, 0], kind='linear')
    dmsc_g_func = sc.interp2d(np.array(range(width)), np.array(range(height)), chsn_img[:, :, 1], kind='linear')
    dmsc_b_func = sc.interp2d(np.array(range(width)), np.array(range(height)), chsn_img[:, :, 2], kind='linear')

    dmsc_img_r = dmsc_r_func(np.arange(0, width, 0.5), np.arange(0, height, 0.5))
    dmsc_img_g = dmsc_g_func(np.arange(0, width, 0.5), np.arange(0, height, 0.5))
    dmsc_img_b = dmsc_b_func(np.arange(0, width, 0.5), np.arange(0, height, 0.5))

    dmsc_img = np.dstack((dmsc_img_r, dmsc_img_g, dmsc_img_b))
    
    #adobe_cuff defined cam values for Nikon D3400 DSLR values fixed with dividing 10000
    adobe_cuff_xyz = np.array([[6988,-1384,-714],
                        [-5631,13410,2447],
                        [-1485,2204,7318]])/10000

    #Given RGB standarts for color spaces 
    sRGB_std =np.array( [[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])

    # Normalize rows to sum to 1 as described in the pdf
    sRGB_to_cam = np.matmul(adobe_cuff_xyz, sRGB_std) 

    #sRGB_to_cam = normalize(sRGB_to_cam, axis=1, norm='l1')
    rgb_sum = np.sum(sRGB_to_cam, axis=1)

    sRGB_to_cam[0, :] = sRGB_to_cam[0, :] / rgb_sum[0]    
    sRGB_to_cam[1, :] = sRGB_to_cam[1, :] / rgb_sum[1]
    sRGB_to_cam[2, :] = sRGB_to_cam[2, :] / rgb_sum[2] 

    #Creating final image matrix multiplication the previous demosaicing with the sRGBtoCam matrix inverse transpose 
    img_csc = np.matmul(dmsc_img,((np.linalg.inv(sRGB_to_cam)).T))
    test=copy.deepcopy(img_csc)
    scl = 2.75
    gryscl_img = sk.color.rgb2gray(scl * test)
    print(np.mean(gryscl_img))
    test = np.clip((scl * test), 0, 1)
    s_e_values = test <= 0.0031308
    bigger_values = test > 0.0031308

    img_result =( s_e_values*test*12.92
                 + np.power((bigger_values*test),1/2.4) * (1 + 0.055)- bigger_values*0.055)
    string= 'result'+ str(typ)+ '.png'
    sk.io.imsave(string, img_result)


    
    
        


# In[69]:


def create_under_birghtness(rgb_picture,typ,scl):
    img_rgb = image_rgb_after_bayer

    #Avarage and Max Values for algorithms
    r_avg = np.mean(img_rgb[:, :, 0])
    r_max = np.max(img_rgb[:, :, 0])

    g_avg = np.mean(img_rgb[:, :, 1])
    g_max = np.max(img_rgb[:, :, 1])

    b_avg = np.mean(img_rgb[:, :, 2])
    b_max = np.max(img_rgb[:, :, 2])
    
    if (typ=='p'):
        r_scale = 2.393118
        g_scale = 1.0
        b_scale = 1.223981
        img_after_pre = img_rgb
        img_after_pre[:, :, 0] = img_after_pre[:, :, 0] * r_scale
        img_after_pre[:, :, 1] = img_after_pre[:, :, 1] * g_scale
        img_after_pre[:, :, 2] = img_after_pre[:, :, 2] * b_scale
        image=img_after_pre
        print('preset_balance')
    
        
    elif (typ=='w'):
        whtwrld_matrix = np.array([[(g_max / r_max), 0, 0],
             [0, 1, 0],
             [0, 0, (g_max / b_max)]])
        image=np.matmul(img_rgb, whtwrld_matrix)
        print('white_balance')
    else:
        grywrld_matrix = np.array([[(g_avg / r_avg), 0, 0],
                             [0, 1, 0],
                             [0, 0, (g_avg / b_avg)]])
        image=np.dot(img_rgb, grywrld_matrix)
        print('gray_balance')
    
    plt.imshow(image)
    chnn_img=image
    height = np.shape(chsn_img)[0]
    width = np.shape(chsn_img)[1]


    dmsc_r_func = sc.interp2d(np.array(range(width)), np.array(range(height)), chsn_img[:, :, 0], kind='linear')
    dmsc_g_func = sc.interp2d(np.array(range(width)), np.array(range(height)), chsn_img[:, :, 1], kind='linear')
    dmsc_b_func = sc.interp2d(np.array(range(width)), np.array(range(height)), chsn_img[:, :, 2], kind='linear')

    dmsc_img_r = dmsc_r_func(np.arange(0, width, 0.5), np.arange(0, height, 0.5))
    dmsc_img_g = dmsc_g_func(np.arange(0, width, 0.5), np.arange(0, height, 0.5))
    dmsc_img_b = dmsc_b_func(np.arange(0, width, 0.5), np.arange(0, height, 0.5))

    dmsc_img = np.dstack((dmsc_img_r, dmsc_img_g, dmsc_img_b))
    
    #adobe_cuff defined cam values for Nikon D3400 DSLR values fixed with dividing 10000
    adobe_cuff_xyz = np.array([[6988,-1384,-714],
                        [-5631,13410,2447],
                        [-1485,2204,7318]])/10000

    #Given RGB standarts for color spaces 
    sRGB_std =np.array( [[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])

    # Normalize rows to sum to 1 as described in the pdf
    sRGB_to_cam = np.matmul(adobe_cuff_xyz, sRGB_std) 

    #sRGB_to_cam = normalize(sRGB_to_cam, axis=1, norm='l1')
    rgb_sum = np.sum(sRGB_to_cam, axis=1)

    sRGB_to_cam[0, :] = sRGB_to_cam[0, :] / rgb_sum[0]    
    sRGB_to_cam[1, :] = sRGB_to_cam[1, :] / rgb_sum[1]
    sRGB_to_cam[2, :] = sRGB_to_cam[2, :] / rgb_sum[2] 

    #Creating final image matrix multiplication the previous demosaicing with the sRGBtoCam matrix inverse transpose 
    img_csc = np.matmul(dmsc_img,((np.linalg.inv(sRGB_to_cam)).T))
    test=copy.deepcopy(img_csc)
    scl = scl
    gryscl_img = sk.color.rgb2gray(scl * test)
    print(np.mean(gryscl_img))
    test = np.clip((scl * test), 0, 1)
    s_e_values = test <= 0.0031308
    bigger_values = test > 0.0031308

    img_result =( s_e_values*test*12.92
                 + np.power((bigger_values*test),1/2.4) * (1 + 0.055)- bigger_values*0.055)
    string= 'result'+ str(typ)+ '.png'
    sk.io.imsave(string, img_result)


# In[ ]:




