import numpy as np
import time
from skimage.io import imread, imsave
import sys


PATH_TO_FILE = ""
DEPTH = None
MSE = 0
sample_img = None
IMAGE = None
quantised_img = None


def handle_arguments():
    global PATH_TO_FILE, DEPTH
    PATH_TO_FILE, DEPTH = sys.argv[1], int(sys.argv[2])

    if DEPTH < 0:
        sys.exit("Depth should be non-negative, your K value is {}".format(DEPTH))


def read_image():
    global sample_img, IMAGE, quantised_img
    sample_img = imread(PATH_TO_FILE).astype("int")
    IMAGE = np.copy(sample_img)
    quantised_img = np.copy(sample_img)
    # print(sample_img.shape)


def median_cut_quantize(img, img_arr, centre):
    global MSE
    # when it reaches the end, color quantize
    # print("to quantize: ", len(img_arr))
    r_average = centre[0]
    g_average = centre[1]
    b_average = centre[2]

    for data in img_arr:
        quantised_img[data[3]][data[4]] = [r_average, g_average, b_average]


def find_centre(img, img_arr):
    # when it reaches the end, color quantize

    r_average = np.mean(img_arr[:, 0])
    g_average = np.mean(img_arr[:, 1])
    b_average = np.mean(img_arr[:, 2])

    return np.array([r_average, g_average, b_average])


def find_dist(img, img_arr, centre):
    output = np.zeros(img_arr.shape[0])
    for idx, data in enumerate(img_arr):
        output[idx] = np.linalg.norm(centre - img[data[3]][data[4]])
    return output


def competitive_learning(img, img_arr, centre, alpha=1, max_iter=5):
    curr_centre = centre
    for itr in range(max_iter):
        alpha = 1 / (itr + 1)
        output = find_dist(img, img_arr, curr_centre)
        idx_winner = output.argmin()
        data = img_arr[idx_winner]
        delta = alpha * (img[data[3]][data[4]] - curr_centre)
        curr_centre = curr_centre + delta
    return curr_centre


def hcl(img, img_arr, depth, centre=None):

    if len(img_arr) == 0:
        return

    if depth == 0:
        median_cut_quantize(img, img_arr, centre)
        return

    if centre is None:
        centre = np.random.randint(255, size=3)
        centre = competitive_learning(img, img_arr, centre)
        # print("Centre Shape: ", centre.shape, centre)

    r_range = np.max(img_arr[:, 0]) - np.min(img_arr[:, 0])
    g_range = np.max(img_arr[:, 1]) - np.min(img_arr[:, 1])
    b_range = np.max(img_arr[:, 2]) - np.min(img_arr[:, 2])

    space_with_highest_range = 0

    if g_range >= r_range and g_range >= b_range:
        space_with_highest_range = 1
    elif b_range >= r_range and b_range >= g_range:
        space_with_highest_range = 2
    elif r_range >= b_range and r_range >= g_range:
        space_with_highest_range = 0

    # print("space_with_highest_range:",space_with_highest_range)

    # sort the image pixels by color space with highest range
    # and find the median and divide the array.
    img_arr = img_arr[img_arr[:, space_with_highest_range].argsort()]
    median_index = int((len(img_arr) + 1) / 2)
    # print("median_index:", median_index)

    # find next centre using median split, followed by CL
    centre1 = find_centre(img, img_arr[0:median_index])
    centre1 = competitive_learning(img, img_arr, (centre1 + centre) / 2)

    # find next centre using median split, followed by CL
    centre2 = find_centre(img, img_arr[median_index:])
    centre2 = competitive_learning(img, img_arr, (centre2 + centre) / 2)

    img_arr1, img_arr2 = [], []

    for idx, data in enumerate(img_arr):
        dist1 = np.linalg.norm(centre1 - data[:3])
        dist2 = np.linalg.norm(centre2 - data[:3])
        if dist1 < dist2:
            img_arr1.append(data.tolist())
        else:
            img_arr2.append(data.tolist())

    img_arr1 = np.array(img_arr1)
    img_arr2 = np.array(img_arr2)

    hcl(img, img_arr1, depth - 1, centre1)
    hcl(img, img_arr2, depth - 1, centre2)


if __name__ == "__main__":

    handle_arguments()

    tick = time.time()
    read_image()

    flattened_img_array = []
    for rindex, rows in enumerate(sample_img):
        for cindex, color in enumerate(rows):
            flattened_img_array.append([color[0], color[1], color[2], rindex, cindex])

    flattened_img_array = np.array(flattened_img_array)
    # print(flattened_img_array.shape)

    # the 3rd parameter represents how many colors are needed in the power of 2. If the parameter
    # passed is 4 its means 2^4 = 16 colors
    tick = time.time()
    hcl(sample_img, flattened_img_array, DEPTH)


    s = set()
    s1 = set()

    for i in range(quantised_img.shape[0]):
        for j in range((quantised_img.shape[1])):
            # print(quantised_img[i][j] - IMAGE[i][j])
            MSE += (np.linalg.norm(IMAGE[i][j] - quantised_img[i][j])) ** 2
            s.add(tuple(quantised_img[i][j]))
            s1.add(tuple(IMAGE[i][j]))

    MSE /= quantised_img.shape[0] * quantised_img.shape[1]

    print()
    print("MSE          :  {:.2f}".format(MSE))
    print("Final Colors : ", len(s))
    print("Time         :  {:.2f}".format(time.time() - tick))
    imsave(PATH_TO_FILE + "_hcl_" + str(2**DEPTH) + ".jpg", quantised_img.astype("uint8"))
    # print("{:.0f}".format(MSE), end = ',')

