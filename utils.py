from PIL import Image
import cv2


def keep_image_size_open(path, size=(512, 512)):
    image=cv2.imread(path)
    # cv2.imshow("Image",image)

    img = Image.open(path)
    # temp = max(img.size)
    # mask = Image.new('P', (temp, temp))
    # mask.paste(img, (0, 0))
    mask = img.resize(size)

    return mask
def keep_image_size_open_rgb(path, size=(512, 512)):
    image = cv2.imread(path)
    # cv2.imshow("Image", image)
    img = Image.open(path)
    # temp = max(img.size)
    # mask = Image.new('RGB', (temp, temp))
    # mask.paste(img, (0, 0))
    mask = img.resize(size)
    return mask
