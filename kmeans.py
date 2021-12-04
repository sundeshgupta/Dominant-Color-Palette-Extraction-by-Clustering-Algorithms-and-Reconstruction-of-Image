from PIL import Image
from matplotlib import pyplot as plt
import sys
import numpy as np
import math
import time

RUN_MODE = -1
PATH_TO_FILE = ""
K = -1
IMAGE = []
IMAGE_3D_MATRIX = []
ORIGINAL_IMAGE_3D_MATRIX = []
MSE = 0
ITR = 20


def kmeans_main(cluster_points):
    # rounding pixel values and getting cluster RGB
    centers = []
    for i in range(len(cluster_points)):
        cluster_points[i] = (
            int(math.floor(cluster_points[i][0])),
            int(math.floor(cluster_points[i][1])),
        )
        red = IMAGE_3D_MATRIX[cluster_points[i][0]][cluster_points[i][1]][0]
        green = IMAGE_3D_MATRIX[cluster_points[i][0]][cluster_points[i][1]][1]
        blue = IMAGE_3D_MATRIX[cluster_points[i][0]][cluster_points[i][1]][2]
        centers.append([red, blue, green])

    centers = np.array(centers)

    # Initializing class and distance arrays
    classes = np.zeros(
        [IMAGE_3D_MATRIX.shape[0], IMAGE_3D_MATRIX.shape[1]], dtype=np.float64
    )
    distances = np.zeros(
        [IMAGE_3D_MATRIX.shape[0], IMAGE_3D_MATRIX.shape[1], K], dtype=np.float64
    )

    for i in range(ITR):
        # finding distances for each center
        for j in range(K):
            distances[:, :, j] = np.sqrt(
                ((IMAGE_3D_MATRIX - centers[j]) ** 2).sum(axis=2)
            )

        # choosing the minimum distance class for each pixel
        classes = np.argmin(distances, axis=2)

        # rearranging centers
        for c in range(K):
            centers[c] = np.mean(IMAGE_3D_MATRIX[classes == c], 0)

    # changing values with respect to class centers
    for i in range(IMAGE_3D_MATRIX.shape[0]):
        for j in range(IMAGE_3D_MATRIX.shape[1]):
            IMAGE_3D_MATRIX[i][j] = centers[classes[i][j]]


def kmeans_with_random():
    global IMAGE_3D_MATRIX
    points = []
    for i in range(K):
        x = np.random.uniform(0, IMAGE_3D_MATRIX.shape[0])
        y = np.random.uniform(0, IMAGE_3D_MATRIX.shape[1])
        points.append((x, y))

    kmeans_main(points)


def read_image():
    global IMAGE, IMAGE_3D_MATRIX, ORIGINAL_IMAGE_3D_MATRIX, ITR
    IMAGE = Image.open(open(PATH_TO_FILE, "rb"))
    IMAGE_3D_MATRIX = np.array(IMAGE).astype(int)
    ORIGINAL_IMAGE_3D_MATRIX = np.array(IMAGE).astype(int)


def handle_arguments():
    global PATH_TO_FILE, K
    PATH_TO_FILE, K = sys.argv[1], int(sys.argv[2])

    if K < 1:
        sys.exit("K should be greater than 1, your K value is {}".format(K))


def save_image():
    global IMAGE_3D_MATRIX, MSE
    im = Image.fromarray(IMAGE_3D_MATRIX.astype("uint8"))
    MSE += (np.linalg.norm(ORIGINAL_IMAGE_3D_MATRIX - IMAGE_3D_MATRIX)) ** 2
    MSE = MSE/(IMAGE_3D_MATRIX.shape[0] * IMAGE_3D_MATRIX.shape[1] * 0.2)

    print()
    print("MSE          :  {:.2f}".format(MSE))
    im.save(PATH_TO_FILE + "_kmeans_" + str(K) + ".jpg")

    s = set()
    s1 = set()
    for i in range(IMAGE_3D_MATRIX.shape[0]):
        for j in range((IMAGE_3D_MATRIX.shape[1])):
            # print(IMAGE_3D_MATRIX[i][j] - ORIGINAL_IMAGE_3D_MATRIX[i][j])
            s.add(tuple(IMAGE_3D_MATRIX[i][j]))
            s1.add(tuple(ORIGINAL_IMAGE_3D_MATRIX[i][j]))
    print("Final Colors : ", K)


if __name__ == "__main__":
    handle_arguments()
    tick = time.time()
    read_image()

    kmeans_with_random()

    save_image()
    # print("{:.2f}".format(MSE), end = ',')

    print("Time         :  {:.2f}".format(time.time() - tick))
