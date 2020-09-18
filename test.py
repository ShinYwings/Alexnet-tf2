import cv2
from matplotlib import pyplot as plt

image = cv2.imread("test.JPEG")
image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
IMAGENET_MEAN = [122.10927936917298, 116.5416959998387, 102.61744377213829]

print(image)
image = image - IMAGENET_MEAN

print(image)
plt.imshow(image)
plt.show()