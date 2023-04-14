import cv2 as cv
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

path = "img/test.jpg"
img = cv.imread(path)
img = img[:500, :500, :]

h, w = img.shape[0:2]

res1 = cv.resize(img, (2*w, 2*h))
res2 = cv.resize(img, None, fx=0.5, fy=0.5)

cv.imshow("ori", img)
cv.imshow("res1", res1)
cv.imshow("res2", res2)

cv.waitKey(0)
cv.destroyAllWindows()


# PIL --> tensor
# PIL: H, W, C  tensor: C, H, W
img = Image.open(path)
img = np.array(img).transpose(2, 0, 1)
img = torch.tensor(img/255)
tran = transforms.Compose([transforms.ToTensor()])
img = tran(img)

# tensor --> picture
img = img.numpy().transpose(1, 2, 0)
img = img * 255
img = Image.fromarray(np.uint8(img))   # 1.
img = Image.fromarray(img.astype(np.uint8))   # 2.


