from __future__ import print_function
from PIL import Image, ImageFilter
import numpy as np
import cv2
from scipy import ndimage
from skimage import exposure, io
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.filters import sobel
from skimage.measure import find_contours
import networkx as nx
from functions import *



def image_brightness():
    # Opent the image
    image = Image.open('image.png')
    
    # Convert the image to a luminous image
    luminous_image = image.point(lambda x: x*1.5)
    
    # Display the image
    luminous_image.show()


def image_acquisition_representation():
    # Open the image
    with Image.open('image.png') as image:
        # Convert the image to numpy array
        image_array = np.array(image)
    
    # Print the shape of the array
    print(image_array.shape)

    #Convert the NumPy array back to an image
    image = Image.fromarray(image_array)
    image.show()


def image_sampling_quantisation():
    # Load the image
    image = io.imread('image.png')
    
    # Select every other pixel in both the rows and columns
    image_sampled = image [::2, ::2]
    
    # Quantize the image by mapping each pixel value to one of 16 discrete values
    image_quantized = np.digitize(image_sampled, np.linspace(0,255,17))
    
    # Display the quantized image
    io.imshow(image_quantized)
    io.show()



def spatial_resolution():
    # Load the image using PIL
    image = Image.open('image.png')

    # Reduce the spatial resolution by a factor of 2
    image_low_res = image.resize((image.width // 3, image.height // 3), resample=Image.BICUBIC)

    # Display the low resolution image
    image_low_res.show()


def intensity_level_resolution():
    # Load the image
    image = Image.open('image.png')

    # Convert the image to grayscale
    im_gray = image.convert('L')

    # Set the intensity level resolution (e.g. 16 levels)
    resolution = 16

    # Create a new image with the desired resolution
    im_res = Image.new(mode='L', size=image.size, color=0)

    # Iterate over the pixels in the image and set their intensity levels
    for x in range(image.width):
        for y in range(image.height):
            # Get the intensity of the current pixel
            intensity = im_gray.getpixel((x, y))

            # Calculate the new intensity level based on the resolution
            new_intensity = intensity // (256 // resolution) * (256 // resolution)

            # Set the new intensity level for the pixel
            im_res.putpixel((x, y), new_intensity)

    # Display the modified image
    im_res.show()


def interpolation(new_width, new_height):
    # Load the image
    image = Image.open("image.png")

    # Resize the image using bilinear interpolation
    new_image = image.resize((new_width, new_height), Image.BILINEAR)

    # Display the new image
    new_image.show()


def contrast_stretching_enchancement():
    # Load the image
    img = cv2.imread("image.png")

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the minimum and maximum intensity values
    min_intensity, max_intensity, _, _ = cv2.minMaxLoc(gray)

    # Stretch the intensity values to the full range
    enhanced_img = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #cv2.imwrite('Contrast_stretching_Enhanced Image.png', enhanced_img)

    # Resize the image
    enhanced_img = cv2.resize(enhanced_img,(1200,700))
    gray = cv2.resize(gray,(1200,700))
    
    # Show the original and enhanced images
    cv2.imshow('Original Image', gray)
    cv2.imshow('Contrast_stretching_Enhanced Image', enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gamma_correction():
    # Load the image
    img = cv2.imread("image.png")

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply gamma correction to the image
    gamma = 1.5
    enhanced_img = np.power(gray / 255, gamma)

    #cv2.imwrite('Gamma_correction_Enhanced Image.png', enhanced_img)

    # Resize the image
    enhanced_img = cv2.resize(enhanced_img,(1200,700))
    gray = cv2.resize(gray,(1200,700))

    # Show the original and enhanced images
    cv2.imshow('Original Image', gray)
    cv2.imshow('Gamma_correction_Enhanced Image', enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def histogram_equalization_enchancement():
    # Load the image
    img = cv2.imread("image.png")

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to the image
    enhanced_img = cv2.equalizeHist(gray)

    #cv2.imwrite('Histogram_equalization_Enhanced Image.png', enhanced_img)
    #cv2.imwrite('Original Gray Image.png', gray)

    # Resize the image
    enhanced_img = cv2.resize(enhanced_img,(1200,700))
    gray = cv2.resize(gray,(1200,700))

    # Show the original and enhanced images
    cv2.imshow('Original Image', gray)
    cv2.imshow('Enhanced Image', enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def negative_image():
    # Load the image
    image = Image.open('image.png')

    # Invert the intensity values
    inverted_image = Image.eval(image, lambda x: 255 - x)

    # Display the inverted image
    inverted_image.show()


def threshold():
    # Load the image
    image = Image.open("image.png")

    # Convert the image to grayscale
    image = image.convert('L')

    # Create the binary image using the threshold
    binary_image = image.point(lambda x: 0 if x < 128 else 255, '1')

    # Display the binary image
    binary_image.show()


def logarithmic_transformation():
    # Load the image
    image = Image.open('image.png')

    # Apply a logarithmic transformation to the image
    image_log = image.point(lambda x: 255 * np.log2(1 + x) / np.log2(256))

    # Display the image
    image_log.show()


def power_law_transformations():
    # Load the image
    img = cv2.imread('image.png', 0)
    
    # Apply the transformation
    gamma = 2.0
    lookup_table = np.array([((i / 255.0) ** (1 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    transformed_image = cv2.LUT(img, lookup_table)
    
    # Resize the images
    img = cv2.resize(img, (1600, 900))
    transformed_image = cv2.resize(transformed_image, (1000, 600))
    
    # Show the original and transformed images
    cv2.imshow('Original Image', img)
    cv2.imshow('Transformed Image', transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def intensity_level_slicing():
    # Load the image
    image = Image.open('image.png')

    # Convert the image to grayscale
    gray = image.convert('L')

    # Select the intensity level to slice at
    intensity_level = 128

    # Create a binary mask
    mask = gray.point(lambda x: 255 if x >= intensity_level else 0)

    # Use the mask to slice the image
    sliced_image = gray.copy()
    sliced_image.paste(0, mask)

    # Display the sliced image
    sliced_image.show()


def grey_level_transformation():
    # Load the image
    image = Image.open('image.png').convert('L')

    # Define the contrast stretch function
    def contrast_stretch(intensity):
        return 255 * (intensity / 255) ** 0.5

    # Apply the contrast stretch transformation
    transformed_image = image.point(contrast_stretch)

    # Display the transformed image
    transformed_image.show()


def bit_plane_slicing():
    #Load the image
    img = cv2.imread('image.png',0)
    
    #Get the size of the image
    rows,cols = img.shape
    
    #Create an array to store the bit planes
    bit_planes= []
    
    #Iterate over all the bits (from 0 to 8)
    for bit in range(8):
        #Create an empty array to store the bit plane
        plane = np.zeros((rows,cols), np.uint8)
        
        #Iterate over the pixels in the image
        for i in range(rows):
            for j in range(cols):
                #Extract the bit from the pixel value
                pixel_bit = (img[i,j]>>bit)&1
                #Set the corresponding pixel in the bit plane
                plane[i,j] = pixel_bit*255
        
        #Add the bit plane to the list
        bit_planes.append(plane)
        
    # Display the bit planes
    for i,plane in enumerate(bit_planes):
        # Resize the image
        plane = cv2.resize(plane,(1200,700))
        cv2.imshow(f'Bit plane {i}',plane)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def spatial_filtering():
    # Open the image
    image = Image.open('image.png')

    # Convert the image to grayscale
    image = image.convert('L')

    # Create a 3x3 matrix for a simple edge detection filter
    matrix = (0, -1, 0,
              -1, 4, -1,
              0, -1, -1)

    # Create the filter
    filter = ImageFilter.Kernel((3,3),matrix)

    # Apply the filter to the image
    image = image.filter(filter)

    # Apply the filter
    image = image.filter(ImageFilter.SMOOTH)

    # Display the image
    image.show()


def linear_spatial_filter():
    # Open the input image
    image = Image.open('image.png')

    # Convert the image to grayscale
    image = image.convert('L')

    # Get the width and height of the image
    width, height = image.size

    # Create a new image with the same size
    filtered_image = Image.new('L', (width, height))

    # Define the filter kernel
    kernel = [[-1, -1, -1],
              [-1, 8, -1],
              [-1, -1, -1]]

    # Get the size of the kernel
    kernel_width = len(kernel[0])
    kernel_height = len(kernel)

    # Calculate the offset for the kernel
    offset = (kernel_width - 1) // 2

    # Iterate over the pixels in the image
    for x in range(offset, width - offset):
        for y in range(offset, height - offset):
            # Apply the filter to the pixel
            filtered_pixel = 0
            for i in range(-offset, offset+1):
                for j in range(-offset, offset+1):
                    filtered_pixel += image.getpixel((x+i, y+j)) * kernel[i+offset][j+offset]
            # Set the pixel in the output image
            filtered_image.putpixel((x, y), filtered_pixel)

    # display the filtered image
    filtered_image.show()


def non_linear_spatial_filter(kernel_size):
    # Load image
    image = cv2.imread("image.png")

    # Create padded image
    padded = cv2.copyMakeBorder(image, kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2, cv2.BORDER_REPLICATE)

    # Create result image
    result = np.zeros_like(image)

    # Loop over image and apply median filter
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract neighborhood
            neighborhood = padded[i:i+kernel_size, j:j+kernel_size]

            # Compute median of neighborhood
            median = np.median(neighborhood)

            # Set result pixel value
            result[i, j] = median

    # Resize the images
    result = cv2.resize(result, (1000, 600))

    # Display result
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def smoothing_spatial_filters():
    # Open the image
    image = Image.open("image.png")

    # Apply the Box blur filter
    image = image.filter(ImageFilter.BoxBlur(radius=4))

    # Show the filtered image
    image.show()


def lowpass_gaussian_filter_kernels():
    # Open the image
    image = Image.open("image.png")

    # Apply the Gaussian blur filter
    image = image.filter(ImageFilter.GaussianBlur(radius=4))

    # Show the filtered image
    image.show()


def non_linear_filters():
    # Open the image and convert it to a numpy array
    image = Image.open("image.png")
    np_im = np.array(image)

    # Apply the minimum filter
    filteredImage = ndimage.minimum_filter(np_im, size=5)

    # Convert the filtered image back to a PIL Image and show it
    filteredImage = Image.fromarray(filteredImage)
    filteredImage.show()


def median_Filter():
    # Open the image
    image = Image.open("image.png")

    # Apply the Median filter
    image = image.filter(ImageFilter.MedianFilter(size=5))

    # Show the filtered image
    image.show()


def weighted_smoothing_filters():
    # Open the image file
    image = Image.open("image.png")

    # Create a 3x3 kernel with weights
    kernel = ImageFilter.Kernel((3, 3), [1, 2, 1,
                                         2, 4, 2,
                                         1, 2, 1])

    # Apply the kernel to the image
    smoothedImage = image.filter(kernel)

    # Show the filtered image
    smoothedImage.show()


def sharpening_spatial_filters_first_derivative():
    # Open the image
    image = Image.open("image.png")

    # Apply the First Derivative filter
    imSharp = image.filter(ImageFilter.EMBOSS)

    # Show the sharpened image
    imSharp.show()

def sharpen_spatial_filter_second_derivative():
    # Open the image
    image = Image.open("image.png")

    # Apply the Second Derivative filter
    imSharp = image.filter(ImageFilter.FIND_EDGES)

    # Show the sharpened image
    imSharp.show()


def second_order_derivative_laplacian():
    # Open the image and convert it to grayscale
    image = Image.open("image.png").convert('L')

    # Convert the image to a NumPy array
    im_array = np.array(image)

    # Create a Laplacian kernel
    kernel = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]])

    # Convolve the image with the kernel
    imageLaplacian = ndimage.convolve(im_array, kernel, mode='reflect')

    # Convert the result back to an image and show it
    imageLaplacian = Image.fromarray(imageLaplacian)
    imageLaplacian.show()


def laplacian_image_enhancement():
    from PIL import Image, ImageFilter
    # Open the image
    image = Image.open('image.png')

    # Apply the Laplacian filter
    imageSharp = image.filter(ImageFilter.Kernel((3, 3), [0, -1, 0,
                                                          -1, 5, -1,
                                                          0, -1, 0], 1))

    # Show the sharpened image
    imageSharp.show()


def histogram_equalization():
    # Load the image as a numpy array
    image = np.array(plt.imread('image.png'))

    # Convert the image to grayscale if it is not already
    image = image.mean(axis=2) if image.ndim == 3 else image

    # Perform histogram equalization
    image_equalized = exposure.equalize_hist(image)

    # Display the image
    plt.imshow(image_equalized, cmap='gray')
    plt.show()


def histogram_matching():
    #Load the images and store them into a variable
    src_image = cv2.imread("image.png")
    ref_image = cv2.imread("reference.jpeg")
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)

    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0, 256])
    src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0, 256])
    src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0, 256])
    ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0, 256])
    ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0, 256])
    ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0, 256])

    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)

    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)

    # Use the lookup function to transform the colors of the original
    # source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)

    # Put the image back together
    image_after_matching = cv2.merge([
        blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)

    #cv2.imwrite('Output Image.png', output_image)

    # Resize the images
    src_image = cv2.resize(src_image, (1200, 700))
    image_after_matching = cv2.resize(image_after_matching, (1200, 700))

    # Show the images
    cv2.imshow('Source Image', src_image)
    cv2.imshow('Reference Image', ref_image)
    cv2.imshow('Output Image', image_after_matching)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contrast_stretching():
    # Load an image
    image = Image.open('image.png')

    # Convert the image to the L mode(grayscale)
    image = image.convert('L')
    
    # Find the minimum and maximum intensity values
    min_intensity = image.getextrema()[0]
    max_intensity = image.getextrema()[1]
    
    # Map the intensity values to the full range (0 to 255)
    stretched = image.point(lambda x: 255*(x-min_intensity)/(max_intensity-min_intensity))
    
    # Display the original and stretched images
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Stretched')
    plt.imshow(stretched, cmap='gray')
    plt.axis('off')
    plt.show()


def image_segmentation():
    # Load an image
    image=Image.open('image.png')
    
    # Convert the image to grayscale
    image_gray = image.convert('L')
    
    # Threshold the image to create a binary image
    threshold = 128
    image_binary = image_gray.point(lambda x: 0 if x < threshold else 255)
    
    # Display the image
    plt.imshow(image_binary, cmap='gray')
    plt.show()


def edge_detection_roberts():
    roberts_cross_v = np.array( [[1, 0 ],[0,-1 ]] )
    roberts_cross_h = np.array( [[ 0, 1 ],[ -1, 0 ]] )
    img = cv2.imread("image.png",0).astype('float64')
    img/=255.0
    vertical = ndimage.convolve( img, roberts_cross_v )
    horizontal = ndimage.convolve( img, roberts_cross_h )
    edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
    edged_img*=255
    
    # Display the image
    plt.imshow(edged_img, cmap='gray')
    plt.show()


def edge_detection_prewitt():
    # Load the image as a numpy array
    image = np.array(plt.imread('image.png'))

    # Convert the image to grayscale if it is not already
    image = image.mean(axis=2) if image.ndim == 3 else image

    # Create the kernels for the Prewitt operator
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Convolve the kernels with the image using valid padding
    edges_x = np.abs(convolve2d(image, kernel_x, mode='valid', boundary='fill', fillvalue=0))
    edges_y = np.abs(convolve2d(image, kernel_y, mode='valid', boundary='fill', fillvalue=0))

    # Combine the horizontal and vertical edges
    edges = np.sqrt(edges_x**2 + edges_y**2)

    # Normalize the edge map
    edges = (edges - edges.min()) / (edges.max() - edges.min())

    # Display the edge map
    plt.imshow(edges, cmap='gray')
    plt.show()


def edge_detection_sobel():
    # Load the image as a numpy array
    image = np.array(plt.imread('image.png'))

    # Perform Sobel edge detection
    edges = sobel(image)

    # Display the edge map
    plt.imshow(edges, cmap='gray')
    plt.show()


def global_thresholding():

    # Load the image
    image = cv2.imread("image.png", 0)

    # Set a threshold value
    threshold_value = 128

    # Threshold the image
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    #cv2.imwrite('Global_Thresholded Image.png', thresholded_image)

    # Resizing the image
    thresholded_image = cv2.resize(thresholded_image, (1200, 700))

    # Show the thresholded image
    cv2.imshow('Global_Thresholded Image', thresholded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def variable_thresholding():
    image = cv2.imread("image.png", 0)

    # Apply Otsu's thresholding
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #cv2.imwrite('Variable_Thresholded Image.png', thresholded_image)

    # Resizing the image
    thresholded_image = cv2.resize(thresholded_image, (1200, 700))

    # Show the thresholded image
    cv2.imshow('Variable_Thresholded Image', thresholded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def local_thresholding():
    image = cv2.imread("image.png", 0)

    # Apply adaptive thresholding
    thresholded_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    #cv2.imwrite('Local_Thresholded Image.png', thresholded_image)

    # Resizing the image
    thresholded_image = cv2.resize(thresholded_image, (1200, 700))

    # Show the thresholded image
    cv2.imshow('Local_Thresholded Image', thresholded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dynamic_thresholding():
    # Load the image
    image = cv2.imread("image.png", 0)

    # Set the minimum and maximum threshold values
    min_threshold = 100
    max_threshold = 200

    # Apply thresholding to the image
    thresholded_image = cv2.inRange(image, min_threshold, max_threshold)

    #cv2.imwrite('Dynamic_Thresholded Image.png', thresholded_image)

    # Resizing the image
    thresholded_image = cv2.resize(thresholded_image, (1200, 700))

    # Show the thresholded image
    cv2.imshow('Dynamic_Thresholded Image', thresholded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


def the_marr_hildreth_edge_detector():
    # read in image as grayscale
    image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

    # apply the Laplacian of Gaussian operator
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # apply non-maximum suppression to thin the edges
    laplacian = cv2.dilate(laplacian, None)
    laplacian = cv2.erode(laplacian, None)

    # detect edges using a threshold
    threshold = 10
    laplacian[laplacian > threshold] = 255
    laplacian[laplacian <= threshold] = 0

    # Resize the image
    laplacian = cv2.resize(laplacian, (1200, 700))

    # display the image
    cv2.imshow('image', laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def canny():
    img = cv2.imread('image.png',0)
    edges = cv2.Canny(img,100,200)
    
    # Display the edge map
    plt.imshow(edges, cmap='gray')
    plt.show()


def connected_component():
    # Load the image
    img = cv2.imread("image.png")

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to the image
    threshold, thresh_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find the connected components in the image
    output = cv2.connectedComponents(thresh_img, 4, cv2.CV_32S)

    # Get the results
    num_labels = output[0]
    labels = output[1]

    # Print the results
    print('Total number of connected components:', num_labels)
    print('Labels:\n', labels)

    # Create a copy of the image to draw the labels on
    label_img = img.copy()

    # Loop through all labels
    for label in range(1, num_labels):
        # Extract the bounding box of the component
        rows, cols = np.where(labels == label)
        top_row, bottom_row = min(rows), max(rows)
        left_col, right_col = min(cols), max(cols)
        bbox = (left_col, top_row, right_col, bottom_row)

        # Draw the bounding box on the copy of the image
        cv2.rectangle(label_img, (left_col, top_row), (right_col, bottom_row), (0, 255, 0), 2)

    #cv2.imwrite('Labeled Image.png', label_img)

    # Resize the images
    img = cv2.resize(img, (1200, 700))
    label_img = cv2.resize(label_img, (1200, 700))

    # Show the original image and the image with the labels drawn on it
    cv2.imshow('Original Image', img)
    cv2.imshow('Labeled Image', label_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def connected_component_labeling():
    # Load the image
    img = cv2.imread('image.png', 0)

    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Perform connected component labeling
    output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)

    # Get the results of the labeling
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    # Print the number of connected components
    print(num_labels)

    # Loop through the labels and draw them on the image
    for i in range(num_labels):
        # Draw the label on the image
        x, y, w, h, area = stats[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Print the centroid coordinates
        cx, cy = centroids[i]
        print(f"Centroid {i}: ({cx}, {cy})")

    # Resize the image
    img = cv2.resize(img, (1200, 700))

    # Show the image
    cv2.imshow('Connected Components', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contour_detection():
    # Load the image as a numpy array
    image = np.array(plt.imread('image.png'))

    # Convert the image to grayscale if it is not already
    image = image.mean(axis=2) if image.ndim == 3 else image

    # Invert the image if necessary
    image = (255 - image) if image.max() > image.min() else image

    # Find the contours of the image
    contours = find_contours(image, 0.5)

    # Display the image with the contours overlaid
    plt.imshow(image, cmap='gray')
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)
    plt.show()


def morphological_filtering():
    # Read in the image
    image = cv2.imread("image.png")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to create a binary image
    thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]

    # Create a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Apply morphological opening to the image
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Resize the image
    opening = cv2.resize(opening, (1200, 700))

    # Display the image
    cv2.imshow("Opening", opening)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hough_transform():
    # Read in the image
    image = cv2.imread("image.png")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Run Hough transform on the edge-detected image
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    # Iterate over the lines and draw them on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Resize the image
    image = cv2.resize(image, (1200, 700))

    # Display the image
    cv2.imshow("Hough Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def edge_linking_global():
    # Load an image file
    image = cv2.imread('image.png')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph for each pixel in the image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            G.add_node((x, y))

    # Iterate over the pixels in the image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # If the pixel is an edge pixel
            if edges[x, y] == 255:
                # Add edges to the graph for each of the neighboring pixels that are connected by an edge
                if x > 0 and edges[x-1, y] == 255:
                    G.add_edge((x, y), (x-1, y))
                if x < image.shape[0] - 1 and edges[x+1, y] == 255:
                    G.add_edge((x, y), (x+1, y))
                if y > 0 and edges[x, y-1] == 255:
                    G.add_edge((x, y), (x, y-1))
                if y < image.shape[1] - 1 and edges[x, y+1] == 255:
                    G.add_edge((x, y), (x, y + 1))

    # Use connected-component labeling to identify and label the different components in the graph
    components = nx.connected_components(G)
    # Draw the edges of the different components on the original image
    for component in components:
        for node in component:
            image[node[0], node[1]] = (0, 255, 0)

    # Resize the image
    image = cv2.resize(image, (1200, 700))

    # Display the image
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def edge_linking_local():
    # Load an image file
    image = cv2.imread('image.png')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find the contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # Resize the image
    image = cv2.resize(image, (1200, 700))

    # Display the image
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # image_brightness()
    # image_acquisition_representation()
    # image_sampling_quantisation()
    # spatial_resolution()
    # intensity_level_resolution()
    # interpolation(100,100)
    # image_enhancement()
    # contrast_stretching_enchancement()
    # gamma_correction()
    # histogram_equalization_enchancement()
    # thresholding()
    # logarithmic_transformation()
    # power_law_transformations()
    # intensity_level_slicing()
    # grey_level_transformation()
    # bit_plane_slicing()
    # spatial_filtering()
    # linear_spatial_filter()
    # non_linear_spatial_filter(3)
    # smoothing_spatial_filters()
    # lowpass_gaussian_filter_kernels()
    # non_linear_filters()
    # median_Filter()
    # weighted_smoothing_filters()
    # sharpening_spatial_filters_first_derivative()
    # sharpen_spatial_filter_second_derivative()
    # second_order_derivative_laplacian()
    # laplacian_image_enhancement()
    # histogram_equalization()
    # histogram_matching()
    # contrast_stretching()
    # image_segmentation()
    # edge_detection_roberts()
    # edge_detection_prewitt()
    # edge_detection_sobel()
    # global_thresholding()
    # variable_thresholding()
    # local_thresholding()
    # dynamic_thresholding()
    # the_marr_hildreth_edge_detector()
    # canny()
    # connected_component()
    # connected_component_labeling()
    # contour_detection()
    # morphological_filtering()
    # hough_transform()
    # edge_linking_global()
    # edge_linking_local()
    pass