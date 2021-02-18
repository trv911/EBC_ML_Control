import sys
import cv2
from PIL import Image


def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale


if __name__ == '__main__':
    img = cv2.imread('cats.jpg')
    image = Image.open(img)
    print("%s\t%s" % (img, calculate_brightness(image)))