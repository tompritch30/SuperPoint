import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from superpoint.settings import EXPER_PATH


model_dir = '/path/to/your/model/directory'

# Load the saved model
loaded_model = tf.saved_model.load(model_dir)

# Print available signatures (functions)
print(list(loaded_model.signatures.keys()))

# Get the serving_default function
inference_func = loaded_model.signatures['serving_default']

# Print input and output details
print(inference_func.structured_input_signature)
print(inference_func.structured_outputs)

def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_size)
    img_orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=-1)
    img = img.astype(np.float32) / 255.0

    return img, img_orig

def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map, keep_k_points=1000):
    def select_k_best(points, k):
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[1], keypoints[0], prob], axis=-1)  # Note the swap of x and y for cv2

    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints = keypoints.astype(int)

    desc = descriptor_map[keypoints[:, 1], keypoints[:, 0]]

    keypoints = [cv2.KeyPoint(p[0], p[1], 1) for p in keypoints]

    return keypoints, desc

def visualize_keypoints(image, keypoints):
    output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    cv2.imshow("Keypoints", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect and visualize SuperPoint keypoints in a single image.')
    parser.add_argument('weights_name', type=str, help='Name of the SuperPoint weights directory.')
    parser.add_argument('img_path', type=str, help='Path to the input image.')
    parser.add_argument('--H', type=int, default=480, help='The height in pixels to resize the image to.')
    parser.add_argument('--W', type=int, default=640, help='The width in pixels to resize the image to.')
    args = parser.parse_args()

    img_size = (args.W, args.H)
    weights_root_dir = Path(EXPER_PATH, 'saved_models')
    weights_dir = Path(weights_root_dir, args.weights_name)

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], str(weights_dir))

        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

        img, img_orig = preprocess_image(args.img_path, img_size)
        out = sess.run([output_prob_nms_tensor, output_desc_tensors], feed_dict={input_img_tensor: np.expand_dims(img, 0)})
        keypoint_map = np.squeeze(out[0])
        descriptor_map = np.squeeze(out[1])
        keypoints, _ = extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map)

        visualize_keypoints(img_orig, keypoints)
