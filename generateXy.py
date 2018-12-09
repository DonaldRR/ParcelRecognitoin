from utils import *
from config import *

# Read Image Names
images_names = []
for roots, dirs, files in os.walk(SPLIT_IMG_DIR):
    images_names.extend(dirs)

# Read training CSV
df = pd.read_csv(TRAINING_CSV_PATH)

# Generating Inputs and Labels
X = []
y = []
sub_imgs = []
# pt12s = []
i = 0
print('# Generating X and y ...')
for image_n in tqdm(images_names):

    img = cv2.imread(os.path.join(ORIGIN_IMG_PATH, image_n + '.jpg'))

    for j in range(7):
        xys = df[df['Image-name'] == (image_n + '.jpg')].iloc[0, j * 10 + 1:j * 10 + 10]
        xys = list(xys)

        if xys[0] == 1:
            y.append(j + 1)

            sub_img = four_point_transform(img, np.reshape(a=xys[1:], newshape=(4, 2)))
            pt1, pt2 = [xys[1], xys[2]], [xys[3], xys[4]]
            # pt12s.append([pt1, pt2])
            sub_img = rotate(sub_img, pt1, pt2)
            cv2.imwrite('./data/train_img/{}.jpg'.format(str(i)), sub_img)
            chars = pytesseract.image_to_string(sub_img)
            X.append(chars)
            i+=1
X = [X[i].replace('\n', ' ') for i in range(len(X))]
print('# Save X to {}\n# Save  y to {}'.format(X_SAVE_PATH, y_SAVE_PATH))
np.save(X_SAVE_PATH, X)
np.save(y_SAVE_PATH, y)
