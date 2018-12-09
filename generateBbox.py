from config import *
from utils import *

images_names = []
for roots, dirs, files in os.walk(ROTATED_IMG_DIR):
    images_names.extend(files)

images_names = [os.path.splitext(f)[0] for f in images_names]
images_names = np.unique(images_names)

print('# Genrerating Candidate Bounding Boxes ...')
for image_name in tqdm(images_names):
    # Read original image and box
    orig_image, orig_bbox = getImageBoxes(image_dir=ROTATED_IMG_DIR, image_name=image_name)
    # Rotate image and box
    rotated_image, rotated_bbox = rotateOrthog(image_name, orig_image, orig_bbox)

    rotated_image_ = rotated_image.copy()
    # Paint inner box area to pure white
    painted_image = paintBOX(boxes=rotated_bbox, image=rotated_image_)
    # Extend more candidates
    extended_bbox = generateCandidateBBOX(image=painted_image, boxes=rotated_bbox, num_iteration=15)
    # Save candidate boxes
    saveSplitImages(rotated_image, extended_bbox, SPLIT_IMG_DIR, image_name)
    cv2.imwrite(join(SPLIT_IMG_DIR, image_name)+'.jpg', rotated_image)

print('# Bounding Boxes saved to {}'.format(SPLIT_IMG_DIR))