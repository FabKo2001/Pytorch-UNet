from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

def rescale_image(image):
    min_image = np.min(image) #Mini- und maximaler Grauwert
    max_image = np.max(image)

    image = 255*(image-min_image).astype(np.float32)/(max_image-min_image) #Neuskalierung, sodass Minimaler Wert zu 0 wird und maximaler zu 255
    image = image.astype(np.uint8)

    return image

image = cv.imread(r'/home/fabiankock/PycharmProjects/BachelorarbeitNeuNeu/gealtete khatode mit AlF3  _029.tif', 0)
image = image[:-70]
image = rescale_image(image)
image = cv.resize(image, (768, 509), interpolation=cv.INTER_LINEAR)

cv.imwrite(r'/preedited_images/_029.tif', image)
