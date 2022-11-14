import cv2
import numpy as np
from glob import glob
from random import random

def _concat_images(im1,im2,path, interpolation=cv2.INTER_CUBIC):
    im1 = cv2.imread(im1)
    im2 = cv2.imread(im2)
    w_min = min(im1.shape[1],im2.shape[1])
    im1 = cv2.resize(im1, (w_min, int(im1.shape[0] * w_min / im1.shape[1])), interpolation=interpolation)
    im2 = cv2.resize(im2, (w_min, int(im2.shape[0] * w_min / im2.shape[1])), interpolation=interpolation)
    imf = cv2.vconcat([im1,im2])
    cv2.imwrite(path, imf)

def _concat_mean(im1,im2,path, interpolation=cv2.INTER_CUBIC):
    im2 = cv2.imread(im2)
    w_min = min(im1.shape[1],im2.shape[1])
    im1 = cv2.resize(im1, (w_min, int(im1.shape[0] * w_min / im1.shape[1])), interpolation=interpolation)
    im2 = cv2.resize(im2, (w_min, int(im2.shape[0] * w_min / im2.shape[1])), interpolation=interpolation)
    imf = cv2.vconcat([im1,im2])
    cv2.imwrite(path, imf)

def dominoes_train(c = 1):
    train_0 = glob("mnist_png/training/0/*.png")
    train_1 = glob("mnist_png/training/1/*.png")
    train_car = glob("cifar/train/*automobile.png")
    train_truck = glob("cifar/train/*truck.png")
    train_0 = iter(train_0)
    train_1 = iter(train_1)
    train_car = iter(train_car)
    train_truck = iter(train_truck)

    for i in range(980):
        if random() < c:
            im1 = next(train_0)
            im2 = next(train_car)
            _concat_images(im1,im2,f"dominoes/train_{int(c*100)}/car/{i}.png")

            im1 = next(train_1)
            im2 = next(train_truck)
            _concat_images(im1,im2,f"dominoes/train_{int(c*100)}/truck/{i}.png")
        
        else:
            im1 = next(train_1)
            im2 = next(train_car)
            _concat_images(im1,im2,f"dominoes/train_{int(c*100)}/car/{i}.png")

            im1 = next(train_0)
            im2 = next(train_truck)
            _concat_images(im1,im2,f"dominoes/train_{int(c*100)}/truck/{i}.png")

        


def dominoes_test(c = 0.5):
    test_0 = glob("mnist_png/testing/0/*.png")
    test_1 = glob("mnist_png/testing/1/*.png")
    test_car = glob("cifar/test/*automobile.png")
    test_truck = glob("cifar/test/*truck.png")
    test_0 = iter(test_0)
    test_1 = iter(test_1)
    test_car = iter(test_car)
    test_truck = iter(test_truck)

    for i in range(980):
        if random() < c:
            im1 = next(test_0)
            im2 = next(test_car)
            _concat_images(im1,im2,f"dominoes/test/car/{i}.png")

            im1 = next(test_1)
            im2 = next(test_truck)
            _concat_images(im1,im2,f"dominoes/test/truck/{i}.png")
        
        else:
            im1 = next(test_1)
            im2 = next(test_car)
            _concat_images(im1,im2,f"dominoes/test/car/{i}.png")

            im1 = next(test_0)
            im2 = next(test_truck)
            _concat_images(im1,im2,f"dominoes/test/truck/{i}.png")

def dominoes_core():
    vis = 35*np.ones((32, 32), np.uint8)
    mnist_mean = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    test_car = glob("cifar/test/*automobile.png")
    test_truck = glob("cifar/test/*truck.png")
    test_car = iter(test_car)
    test_truck = iter(test_truck)

    for i in range(980):
        im2 = next(test_car)
        _concat_mean(mnist_mean,im2,f"dominoes/core/car/{i}.png")

        im2 = next(test_truck)
        _concat_mean(mnist_mean,im2,f"dominoes/core/truck/{i}.png")