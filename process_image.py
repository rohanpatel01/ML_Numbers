import os.path
import gzip
import mnist
from mnist import MNIST

mndata = MNIST('samples')
images, labels = mndata.load_training()

imageIndex = 11

def changeArray():
    myPrint = images[imageIndex]

    for x in range(len(myPrint)):
        val = myPrint[x]
        if val != 0:
            myPrint[x] = '@'
        else:
            myPrint[x] = '.'



def squishArray():
    myPrint = images[imageIndex]

    for x in range(len(myPrint)):
        # print("Before: ", myPrint[x])
        myPrint[x] = myPrint[x] / 255
        # print("After: ", myPrint[x])


def myPrint():
    show = images[imageIndex]

    i = 0
    while (i < len(show)):
        for x in range(28):
            print(show[i + x], end="")
        # print(show[i: i + 28])
        print()
        i += 28


# print(images[0])
# or
# images, labels = mndata.load_testing()

def main():
    print(mndata.display(images[imageIndex]))
    print(labels[imageIndex])

    # changeArray()
    # squishArray()
    myPrint()

    print(images[imageIndex])


    print("work")




main()