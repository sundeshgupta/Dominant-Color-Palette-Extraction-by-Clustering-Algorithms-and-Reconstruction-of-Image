import cv2
import numpy as np
import math
import time
import sys

class Image:
    def __init__(self, path, nb=0):
        self.imgpath = path
        self.orig_img = []
        self.colors = []
        self.freq = []
        self.height = 0
        self.width = 0
        self.numcolors = 0
        self.nb = nb
        self.final_img = []
        self.final_img_reshaped = []

    def extract_colors(self):
        img = cv2.imread(self.imgpath).astype("int")
        self.image = img
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.numcolors = img.shape[2]
        self.final_img = np.zeros([self.height, self.width, self.numcolors])
        reshaped = img.reshape((self.height * self.width, self.numcolors))
        self.orig_img = reshaped
        for i in range(len(reshaped)):
            reshaped[i] = (reshaped[i] >> self.nb) << self.nb
        return reshaped

    def generate_pallete(self):
        reshaped = self.extract_colors()
        self.colors, self.freq = np.unique(reshaped, axis=0, return_counts=True)
        count_sort_ind = np.argsort(-self.freq)
        self.colors = self.colors[count_sort_ind]
        self.freq = self.freq[count_sort_ind]

    def write_final_image(self, imgpath="final.jpg"):
        self.final_img_reshaped = self.final_img.reshape(
            (self.height * self.width, self.numcolors)
        )
        twodim_img = self.final_img.astype("uint8")
        cv2.imwrite(imgpath, twodim_img)

    def calc_mse(self):
        err = np.linalg.norm(self.final_img_reshaped - self.orig_img)
        err = err ** 2
        err = err / (self.width * self.height)
        return err

    def check_colors_final(self):
        return np.unique(self.final_img_reshaped, axis=0).shape[0]


def distance(a, b):
    return np.linalg.norm(a - b)


def find_nearest_dominant(hist_bin, dominants):
    min_dist = 16 ** 3
    min_ind = -1
    for i in range(len(dominants)):
        if distance(hist_bin[1], dominants[i][1]) < min_dist:
            min_dist = distance(hist_bin[1], dominants[i][1])
            min_ind = i
    return (min_dist, min_ind)


def adaptive_clustering(Nb, Np, imgpath, classes):
    thresh = math.pow((255 * 255 * 255) / classes, 1.0 / 3)
    flag = 0
    nbins = 0
    bins = []
    img = Image(imgpath, nb=Nb)
    img.generate_pallete()

    for i in range(len(img.freq)):
        if img.freq[i] >= Np:
            bins.append((img.freq[i], img.colors[i]))
            nbins += 1

    neighbours = []
    for i in range(classes):
        neighbours.append([[], []])

    while not flag:
        dominants = []
        flags = []

        dominants.append(bins[0])
        flags.append(1)
        index = 1

        while len(dominants) < classes and index < nbins:
            dist, ind = find_nearest_dominant(bins[index], dominants)

            if dist > thresh:
                dominants.append(bins[index])
                flags.append(1)
            else:
                flags.append(0)

            index += 1

        if len(dominants) < classes:
            unfound = classes - len(dominants)
            if unfound > 64:
                thresh -= 4
            else:
                thresh -= 2
            continue
        else:
            flag = 1

        while index < nbins:
            flags.append(0)
            index += 1

        for i in range(nbins):
            _, ind = find_nearest_dominant(bins[i], dominants)
            neighbours[ind][0].append(bins[i][0])
            neighbours[ind][1].append(bins[i][1])

        for i in range(classes):
            summation = np.array([0, 0, 0])
            nvalues = 0
            dominants[i] = np.around(
                np.average(neighbours[i][1], axis=0, weights=neighbours[i][0])
            )

        classess = np.zeros([img.height, img.width], dtype=np.float64)
        distances = np.zeros([img.height, img.width, classes], dtype=np.float64)

        for j in range(classes):
            distances[:, :, j] = np.sqrt(((img.image - dominants[j]) ** 2).sum(axis=2))

        classess = np.argmin(distances, axis=2)

        for i in range(img.height):
            for j in range(img.width):
                img.final_img[i][j] = dominants[classess[i][j]]

        img.write_final_image(imgpath=imgpath + "_ada_" + str(classes) + ".jpg")

        print()
        # print("{:.2f}".format(img.calc_mse()), end = ',')
        print("MSE          :  {:.2f}".format(img.calc_mse()))
        print("Final Colors : ", img.check_colors_final())


imgpath = sys.argv[1]

Nb = 3

Np = 3

classes = int(sys.argv[2])

start = time.time()

adaptive_clustering(Nb, Np, imgpath, classes)

end = time.time()

print("Time         :  {:.2f}".format(end - start))
