from PIL import Image
import numpy as np


def IntensityLevelSlicing():
    # Load the image
    image = Image.open('image.jpg')

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


def IntensityLevelResolution():
    # Load the image
    im = Image.open('image.jpg')

    # Convert the image to grayscale
    im_gray = im.convert('L')
    im_gray.show()

    # Set the intensity level resolution (e.g. 16 levels)
    resolution = 16

    # Create a new image with the desired resolution
    im_res = Image.new(mode='L', size=im.size, color=0)
    im_res.show()

    # Iterate over the pixels in the image and set their intensity levels
    for x in range(im.width):
        for y in range(im.height):
            # Get the intensity of the current pixel
            intensity = im_gray.getpixel((x, y))

            # Calculate the new intensity level based on the resolution
            new_intensity = intensity // (256 // resolution) * (256 // resolution)

            # Set the new intensity level for the pixel
            im_res.putpixel((x, y), new_intensity)

    # Display the modified image
    im_res.show()


def NegativeImage():
    # Load the image
    image = Image.open('image.jpg')

    # Invert the intensity values
    inverted_image = Image.eval(image, lambda x: 255 - x)

    # Display the inverted image
    inverted_image.show()


def GreylevelTransformation():
    # Load the image
    image = Image.open('image.jpg').convert('L')

    # Define the contrast stretch function
    def contrast_stretch(intensity):
        return 255 * (intensity / 255) ** 0.5

    # Apply the contrast stretch transformation
    transformed_image = image.point(contrast_stretch)

    # Display the transformed image
    transformed_image.show()


def LinearSpatialFilter():
    # Open the input image
    image = Image.open('image.jpg')

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


def LogarithmicTransformation():
    # Load the image
    image = Image.open('image.jpg')

    # Apply a logarithmic transformation to the image
    image_log = image.point(lambda x: np.log(x + 1))

    # Display the image
    image_log.show()