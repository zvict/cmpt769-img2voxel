import cv2
import os


if __name__ == '__main__':
    img_dir = './data'
    depth_dir = 'NYU-Depth'
    dst_dir = 'NYU-Depth'
    dst_rgb_dir = 'NYU-RGB'
    factor = 2

    os.makedirs(os.path.join(img_dir, dst_dir), exist_ok=True)
    os.makedirs(os.path.join(img_dir, dst_rgb_dir), exist_ok=True)

    for img_id in os.listdir(os.path.join(img_dir, 'NYU-RGB')):
        # if img_id.endswith('.png'):
            # img_id = img_id[:4]
        img_id = img_id.split('.')[0]
        print(img_id)
        depth = cv2.imread(os.path.join(img_dir, depth_dir, img_id + '.png'), cv2.IMREAD_ANYDEPTH)
        depth = cv2.resize(depth, (depth.shape[1] // factor, depth.shape[0] // factor))
        cv2.imwrite(os.path.join(img_dir, dst_dir, img_id + '_lowres.png'), depth)

        try:
            img = cv2.imread(os.path.join(img_dir, 'NYU-RGB', img_id + '.png'))
            img = cv2.resize(img, (img.shape[1] // factor, img.shape[0] // factor))
        except:
            img = cv2.imread(os.path.join(img_dir, 'NYU-RGB', img_id + '.jpg'))
            img = cv2.resize(img, (img.shape[1] // factor, img.shape[0] // factor))
        cv2.imwrite(os.path.join(img_dir, dst_rgb_dir, img_id + '_lowres.png'), img)