#!/usr/bin/env python
# coding: utf-8

# In[5]:


def get_image_data(im_dir):
    '''
    INPUT: name of directory containing images (str), with sub-directories separated by '\\'
    OUTPUT: pandas dataframe of the gray-scale image N x 5 array with N = number of images and
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

    # Now create output array
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
    from scipy.ndimage import gaussian_filter
    
    image = img_as_ubyte(image[:,:,0])
    return canny(image,sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)


# In[15]:


def define_circular_roi(image,edge):
    '''
    INPUT: - Plate Image 
           - edge detected by plate_edge_detector
    OUPUT: - region of interest (roi) in which analysis will be done
           - Pixel area of ROI
    modules: skimage, numpy
    '''
    # Imports
    from skimage.transform import hough_circle, hough_circle_peaks
    from skimage.draw import circle
    from numpy import arange, zeros,shape, pi
    
    # Find plate perimeter
    image = image[:,:,0]
    r = 1880/2 
    hough_radii = arange(r-5,r+5, 2)
    hough_res = hough_circle(edge,hough_radii)
    accums, cx, cy, radius = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=1)
    # Subtract edge to improve counting
    radius -= 175
    circy, circx = circle(int(cy), int(cx), int(radius),shape=image.shape)
    
    # crop image to show only the plate
    cropped_image = image[int(cy-radius):int(cy+radius),
                         int(cx-radius):int(cx+radius)]
    roi = zeros(shape(image))
    roi[circy,circx] = 1
    roi_cropped = roi[int(cy-radius):int(cy+radius),
                      int(cx-radius):int(cx+radius)]
    roi_area = pi * radius ** 2
    return cropped_image * roi_cropped, roi_area

def CFU_count(im):
    '''
    INPUT:- plate image array
          -  optional : threshold for detection (b/w 0 and 1) and maximum overlap between cells
    OUTPUT: an Nx3 array of [x,y,are] of each detected blob
    modules: skimage, scipy, numpy
    '''
    
    # Imports
    from skimage.filters import median
    from skimage.feature import blob_log
    from skimage.color import rgb2gray
    from skimage.morphology import reconstruction
    from scipy.ndimage import gaussian_filter
    from numpy import sqrt, pi,copy
    
   
    # Background Subtraction
    image = gaussian_filter(im, 1)
    seed = copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    dilated = reconstruction(seed, mask, method='dilation')
    im_filt = image - dilated
    
    # median filtering 
    im_float = (im_filt.astype(float) - im_filt.min()) / (im_filt.max() - im_filt.min())
    median_filt = median(im_float) > 70
    
    # CFU detection 
    im_grey = rgb2gray(median_filt)
    blobs_log = blob_log(im_grey,min_sigma=6,max_sigma=10,threshold=.4)
    blobs_log[:, 2] = 2 * pi * blobs_log[:, 2] ** 2
        
    return blobs_log

def show_CFUs(cropped_im,blobs_log,figsize=(10,10)):
    '''
    INPUT: - plate image array
           - The 3xN array of all counted colonies
           - optional figure size, (h,w) tuple
    OUTPUT: - matplotlib display of highlighted colonies
    modules: matplotlib,skimage,numpy
    '''
    # Imports
    import matplotlib.pyplot as plt
    from skimage.io import imshow
    from skimage.color import rgb2gray
    from numpy import sqrt, pi
    
    im_grey = rgb2gray(cropped_im)
    fig, ax = plt.subplots(figsize=figsize)
    imshow(im_grey,cmap='gray')
    for blob in blobs_log:
        y, x, a = blob
        r = sqrt(a/pi)
        c = plt.Circle((x,y), r, color = 'red',linewidth=2,fill=False)
        ax.add_artist(c)
    plt.show()