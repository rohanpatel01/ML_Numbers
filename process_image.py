import os.path
import gzip
import mnist

from mnist import MNIST

mndata = MNIST('samples')

images, labels = mndata.load_training()

print(images[0])
# or
# images, labels = mndata.load_testing()

# print(mndata.display(images[0]))

print("work")