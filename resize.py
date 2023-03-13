import cv2
import os


if __name__ == '__main__':
    img_dir = './data'
    depth_dir = 'SI_R20'
    dst_dir = 'SI_R20_lowres'
    dst_rgb_dir = 'RGB_lowres'
    factor = 10

    os.makedirs(os.path.join(img_dir, dst_dir), exist_ok=True)
    os.makedirs(os.path.join(img_dir, dst_rgb_dir), exist_ok=True)

    for img_id in os.listdir(os.path.join(img_dir, 'RGB')):
        if img_id.endswith('.png'):
            img_id = img_id[:4]
            print(img_id)
            depth = cv2.imread(os.path.join(img_dir, depth_dir, img_id + '_final_midas_v2_o2m.png'), cv2.IMREAD_ANYDEPTH)
            depth = cv2.resize(depth, (depth.shape[1] // factor, depth.shape[0] // factor))
            cv2.imwrite(os.path.join(img_dir, dst_dir, img_id + '_final_midas_v2_o2m_lowres.png'), depth)

            img = cv2.imread(os.path.join(img_dir, 'RGB', img_id + '_rgb.png'))
            img = cv2.resize(img, (img.shape[1] // factor, img.shape[0] // factor))
            cv2.imwrite(os.path.join(img_dir, dst_rgb_dir, img_id + '_rgb_lowres.png'), img)