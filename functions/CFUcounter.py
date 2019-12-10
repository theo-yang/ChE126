#!/usr/bin/env python
# coding: utf-8

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
    # Remove wood agar assay pics taken on 12/02/19
    im_list = [ x for x in im_list if "120219" not in x ]
    images = [imread(im_name) for im_name in im_list]
    # Load images and store in list 
    im_list = list(im[im.find('data\\')+5:] for im in im_list)
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

def colony_coordinates(image_path):
    '''
    INPUTS: - Image Path
    OUPUTS: - List of coordinates of user-selected colonies
    Attribution: This function is a modified version of record_clics() from the bebi103 module. See github.com/justinbois/bebi103 for documentation
    Modules: bokeh, skimage, numpy
    
    '''
    # Imports
    import bokeh
    import bokeh.plotting
    from skimage.io import imread
    import numpy as np
    
    
    points_source = bokeh.models.ColumnDataSource({"x": [], "y": []})
    def modify_doc(doc,frame_height=400):
        frame = imread(image_path)
        M, N, _ = frame.shape
        img = np.empty((M, N), dtype=np.uint32)
        view = img.view(dtype=np.uint8).reshape((M, N, 4))
        view[:,:,0] = frame[:,:,0] # copy red channel
        view[:,:,1] = frame[:,:,1] # copy blue channel
        view[:,:,2] = frame[:,:,2] # copy green channel
        view[:,:,3] = 255
        img = img[::-1]
        frame_width = int(frame_height * N/M)
        p = bokeh.plotting.figure(
                frame_height=frame_height,
                frame_width=frame_width,
                tools="pan,box_zoom,wheel_zoom,save,reset")
        p.image_rgba(image=[img],x=0, y=0, dw=frame_width,dh=frame_height)

        view = bokeh.models.CDSView(source=points_source)

        renderer = p.scatter(
            x="x",
            y="y",
            source=points_source,
            view=view,
            color='red',
            size=3,
        )

        columns = [
            bokeh.models.TableColumn(field="x", title="x"),
            bokeh.models.TableColumn(field="y", title="y"),
        ]

        table = bokeh.models.DataTable(
            source=points_source, columns=columns, editable=True, height=200
        )

        draw_tool = bokeh.models.PointDrawTool(renderers=[renderer])
        p.add_tools(draw_tool)
        p.add_tools(bokeh.models.CrosshairTool(line_alpha=.5))
        p.toolbar.active_tap = draw_tool

        doc.add_root(bokeh.layouts.column(p, table))
    
    bokeh.io.show(modify_doc, notebook_url="localhost:8888")
    return points_source

def colony_dists(coordinates,sample_name,dilution):
    '''
    Computes Distances between CFUs and wood/pencil substrates
    INPUT: - Nx2 numpy array of coordinates, where the first two coordinates are wood and pencil shavings respectively
           - sample name (str) and -log2(dilution) (int)
    OUTPUT: Distances between CFU and each shaving as a pandas DataFrame
    modules: numpy, pandas

    '''
    
    # Imports
    import numpy as np
    import pandas as pd
    
    num_points = len(coordinates[:,0])
    num_colonies = num_points - 2
    
    wood = coordinates[0,:]
    pencil = coordinates[1,:]
    
    # Find Distances
    def distance(x):
        return [np.sqrt((x[0] - wood[0]) ** 2 + (x[1] - wood[1]) ** 2),np.sqrt((x[0] - pencil[0]) ** 2 + (x[1] - pencil[1]) ** 2)]
    
    dists = np.apply_along_axis( distance, axis=1, arr=coordinates)
    
    # Distance between shavings
    substrate_dist = distance(wood)[1]
    
    # Assemble DataFrame
    label = np.array([[sample_name]*num_points]).T
    label[0] = 'wood'
    label[1] = 'pencil'
    dilution = np.array([[dilution]*num_points]).T
    
    return pd.DataFrame(np.concatenate((label,dilution,coordinates,dists),axis=1),columns=['sample','-Log2(dilution)','x','y','Dist (wood)','Dist (pencil)'])
    