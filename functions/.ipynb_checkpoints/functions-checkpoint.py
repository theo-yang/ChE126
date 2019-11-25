#!/usr/bin/env python
# coding: utf-8

# In[5]:


def get_image_data(im_dir):
    '''
    INPUT: name of directory containing images (str), with sub-directories separated by '\\'
    OUTPUT: pandas dataframe of the gray-scale image N x 5 matrix with N = number of images and
            columns = [Date, Environ, Choloramphenicol,-Log10(Dilution),gray-scale pixel values]
    MODULES: glob, skimage, pandas
    '''
    
    # Imports
    import glob
    from skimage.io import imread
    from skimage.color import rgb2gray
    import pandas as pd
    
    if im_dir.endswith('.JPG'):
        im_list = [im_dir]
    else:
        im_list = list(glob.glob(im_dir + "/**/*.JPG",recursive=True))
    images = [imread(im_name) for im_name in im_list]
    # Load images and store in list 
    im_list = list(im[im.find('data\\') + 6:] for im in im_list)
    im_list = list(im.split('\\') + [images[idx]] for idx, im in enumerate(im_list))
    images = pd.DataFrame(im_list,columns = ['Date','treatment','Image'])
    
    # remove .JPG tag, make new columns for treatment
    treatment = images.pop('treatment')
    conditions_df = pd.concat([treatment] * 3, axis = 1)
    conditions_df.columns = ['CHL','Environ','-Log(Dilution)']
    for idx, condition in enumerate(treatment):
        agar, environ, dilution = tuple(condition.split('_'))
        if agar == 'MEA':
            conditions_df.loc[idx,'CHL'] = 'No'
        elif agar == 'MEACHL':
            conditions_df.loc[idx,'CHL'] = 'Yes'
        conditions_df.loc[idx,'Environ'] = environ
        conditions_df.loc[idx,'-Log(Dilution)'] =  dilution.replace('.JPG','')

    # Now create output matrix
    data = pd.concat([images,conditions_df], axis = 1)
    data_arranged = data[['Date','Environ','CHL','-Log(Dilution)','Image']]
    
    return data_arranged


# In[9]:


def plate_edge_detector(image,sigma=5,low_threshold=7,high_threshold=40):
    '''
    INPUT: - Plate Image
           - optional: sigma, low threshold, high_theshold (see skimage)
    OUPUT: Canny Filter for edge of plate
    modules: skimage
    '''
    # Imports
    from skimage.feature import canny
    from skimage.util import img_as_ubyte
    
    image = img_as_ubyte(image[:,:,0])
    return canny(image,sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)


# In[15]:


def define_circular_roi(image,edge):
    '''
    INPUT: - Plate Image 
           - edge detected by plate_edge_detector
    OUPUT: region of interest (roi) in which analysis will be done
    modules: skimage, numpy
    '''
    # Imports
    from skimage.transform import hough_circle, hough_circle_peaks
    from skimage.draw import circle
    from numpy import arange, zeros,shape
    image = image[:,:,0]
    r = 1880/2 # pixel radius of plate.
    hough_radii = arange(r-5,r+5, 2)
    hough_res = hough_circle(edge,hough_radii)
    accums, cx, cy, radius = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=1)
    radius = radius - 175
    circy, circx = circle(int(cy), int(cx), int(radius),shape=image.shape)
    
    # crop image to show only the plate
    cropped_image = image[int(cy-radius):int(cy+radius),
                         int(cx-radius):int(cx+radius)]
    roi = zeros(shape(image))
    roi[circy,circx] = 1
    roi_cropped = roi[int(cy-radius):int(cy+radius),
                      int(cx-radius):int(cx+radius)]
    return cropped_image * roi_cropped

def CFU_counter(im):
    '''
    INPUT: plate image matrix
    OUTPUT: (Number of CFUs, Area occupied by colonies)
    modules: skimage, scipy
    '''
    import skimage
    from matplotlib.patches import Circle
    from skimage import data, color
    import scipy
    import matplotlib.pyplot as plt
    from skimage import data, feature, exposure
    from skimage.transform import hough_circle, hough_circle_peaks
    from skimage.draw import circle_perimeter
    from skimage.feature import canny
    from skimage.util import img_as_ubyte
    from numpy import sqrt, arange
    from math import sqrt
    from skimage import data
    from skimage.feature import blob_log
    from skimage.color import rgb2gray
    
    # Bespoke threshold filtering
    im_float = (im.astype(float) - im.min()) / (im.max() - im.min())
    bin_im = skimage.filters.median(im_float) > 140
    im_clean = skimage.morphology.remove_small_holes(bin_im, 100)
    
    im_grey = rgb2gray(im_clean)
    blobs_log = blob_log(im_grey,min_sigma=5,max_sigma=20,threshold=.5,overlap=.2)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    fig, ax = plt.subplots(figsize=(10,10))
    skimage.io.imshow(im_grey,cmap='gray')
    for blob in blobs_log:
        y, x, r = blob
        c = plt.Circle((x,y), r, color = 'red',linewidth=2,fill=False)
        ax.add_artist(c)
    plt.show()
        
    return blobs_log