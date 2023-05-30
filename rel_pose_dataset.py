import os
from pathlib import Path
import msgpack
import msgpack_numpy as m

import numpy as np
import torch

from OwlSam.OwlSAM import OWL_SAM
from segment_table import segment_table

import apriltag
import cv2
from se3_tools import so3_log, rot2rotvec

m.patch()
torch.set_grad_enabled(False)

pose_folder = 2
allignment_axis_error_threshold = 5
show = True
# device = 'cpu'
device = 'cuda:0'

if device == 'cpu':
    print("[INFO] Remember to switch device to 'cuda:0' when possible \n\n")

def pose_inv(pose):
    R = pose[:3, :3]
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = - R.T @ np.ascontiguousarray(pose[:3, 3])
    return T

def show_tag(detection, image):
    # extract the bounding box (x, y)-coordinates for the AprilTag
    # and convert each of the (x, y)-coordinate pairs to integers
    (ptA, ptB, ptC, ptD) = detection.corners
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    ptA = (int(ptA[0]), int(ptA[1]))
    # draw the bounding box of the AprilTag detection
    cv2.line(image, ptA, ptB, (0, 255, 0), 2)
    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
    cv2.line(image, ptC, ptD, (0, 255, 0), 2)
    cv2.line(image, ptD, ptA, (0, 255, 0), 2)
    # draw the center (x, y)-coordinates of the AprilTag
    (cX, cY) = (int(detection.center[0]), int(detection.center[1]))
    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
    # draw the tag family on the image
    tagFamily = detection.tag_family.decode("utf-8")
    cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # show the output image after AprilTag detection
    cv2.imshow("Image", image)
    cv2.waitKey(0)

class Saver():

    def __init__(self, first_scene_id=0) -> None:
        self.scene_id = first_scene_id
        self.output_dir = Path("/home/pita/HDD/sdb/datasets/val_sar")

    def save_file(self, data):
        destination_dir = os.path.join(self.output_dir, 'scene_{:07d}'.format(self.scene_id) + '.msgpack')
        with open(destination_dir, "wb") as outfile:
            packed = msgpack.packb(data)
            outfile.write(packed)
            print(f"Scene {self.scene_id} saved")
        self.scene_id += 1

    def save_data(self, rgb0, rgb1,
                  depth0, depth1,
                  seg0, seg1,
                  K, T_WC, T_W0, T_W1,
                  T_delta_C, T_delta_B):
        #make sure dpeth in m
        colors = [rgb0, rgb1]
        depths = [depth0, depth1]
        segmaps = [seg0, seg1]
        data = {
            'colors': colors,
            'depth': depths,
            'intrinsic': K,
            'T_WC_opencv': T_WC,
            'T_WO_frame_0': T_W0,
            'T_WO_frame_1': T_W1,
            'T_delta_cam': T_delta_C,
            'T_delta_base': T_delta_B,
            'cp_main_obj_segmaps': segmaps,
        }
        self.save_file(data)

    def save_all_data(self, rgb0, rgb1,
                      depth0, depth1,
                      seg0, seg1,
                      K, T_WC, T_C0, T_C1):
        T_W0 = T_WC @ T_C0
        T_W1 = T_WC @ T_C1

        T_delta_C, T_delta_B = get_delta_transformation(T_C0, T_C1)
        self.save_data(
            rgb0, rgb1, depth0, depth1,
            seg0, seg1, K, T_WC, T_W0, T_W1,
            T_delta_C, T_delta_B
        )

        T_delta_C, T_delta_B = get_delta_transformation(T_C1, T_C0)
        self.save_data(
            rgb1, rgb0, depth1, depth0,
            seg1, seg0, K, T_WC, T_W1, T_W0,
            T_delta_C, T_delta_B
        )


images_dir = f"pose_estimation_{pose_folder}"

K_cam = np.load(images_dir + "/head_cam_intrinsic_matrix.npy")
T_WC = np.load(images_dir + "/T_WC_head_individual_calibration.npy")

options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)

segmentator = OWL_SAM(device)
saver = Saver(0)

if pose_folder == 1:
    tag_size = 2.35
elif pose_folder == 2:
    tag_size = 4.55

def get_delta_transformation(T_C0, T_C1):
    T_delta_cam = T_C1 @ pose_inv(T_C0)
    T_delta_base = T_WC @ T_delta_cam @ pose_inv(T_WC)
    return T_delta_cam, T_delta_base

def get_transformations(gray0, bgr0, gray1, bgr1):
    detection = detector.detect(gray0)[0]
    pose = detector.detection_pose(detection, (K_cam[0,0], K_cam[1,1], K_cam[0,2], K_cam[1,2]), tag_size=tag_size, z_sign=1)
    T_C0 = pose[0]
    if show:
        show_tag(detection, bgr0)

    detection = detector.detect(gray1)[0]
    pose = detector.detection_pose(detection, (K_cam[0,0], K_cam[1,1], K_cam[0,2], K_cam[1,2]), tag_size=tag_size, z_sign=1)
    T_C1 = pose[0]
    if show:
        show_tag(detection, bgr1)

    T_delta_cam ,T_delta_base = get_delta_transformation(T_C0, T_C1)

    # print(T_WC @ T_C0)
    # print(T_delta_base)

    rotvec = rot2rotvec(T_delta_base[:3, :3])
    delta_magnitude = np.linalg.norm(rotvec)
    axis = rotvec / delta_magnitude
    z_axis = np.array([0,0,1])
    if np.dot(z_axis, axis) < 0:
        delta_magnitude *= -1
        axis = -axis

    axis_allignment_error = np.rad2deg(np.arccos(np.dot(z_axis, axis)))

    print(f"Rotation: {delta_magnitude * 180/np.pi} \n")

    valid_pair = True
    print_color = "\033[32m"
    if axis_allignment_error > allignment_axis_error_threshold:
        valid_pair = False
        print_color = "\033[31m"
    print(T_delta_base)
    print(f"Delta z-axis: {np.abs(1 - T_delta_base[2,2])}")
    print(f"{print_color}Allignment with z-axis: {axis_allignment_error}\033[0m")
    print_color = "\033[32m"
    if np.abs(delta_magnitude * 180/np.pi) > 45:
        valid_pair = False
        print_color = "\033[31m"
    print(f"{print_color}Rotation: {delta_magnitude * 180/np.pi}\033[0m\n")

    return T_C0, T_C1, valid_pair

def get_data_ordered_list(id_range, text_prompt):
    text_prompt = 'An image of a ' + text_prompt

    for i in range(0, len(id_range)-2, 2):
        print(f"[INFO] Processing the pair {i/2+1} of {text_prompt}")

        image_name = f"head_camera_rgb_{id_range[i]}.png"
        bgr0_tag = cv2.imread(images_dir + "/" + image_name, )
        gray0_tag = cv2.cvtColor(bgr0_tag, cv2.COLOR_BGR2GRAY)

        image_name = f"head_camera_rgb_{id_range[i+1]}.png"
        bgr0 = cv2.imread(images_dir + "/" + image_name, )
        rgb0 = bgr0.copy()[:,:,[2,1,0]]

        image_name = f"head_camera_rgb_{id_range[i+2]}.png"
        bgr1_tag = cv2.imread(images_dir + "/" + image_name, )
        gray1_tag = cv2.cvtColor(bgr1_tag, cv2.COLOR_BGR2GRAY)

        image_name = f"head_camera_rgb_{id_range[i+3]}.png"
        bgr1 = cv2.imread(images_dir + "/" + image_name, )
        rgb1 = bgr1.copy()[:,:,[2,1,0]]

        image_name = f"head_camera_depth_to_rgb_{id_range[i+1]}.png"
        depth0 = cv2.imread(images_dir + "/" + image_name, cv2.IMREAD_UNCHANGED)
        image_name = f"head_camera_depth_to_rgb_{id_range[i+3]}.png"
        depth1 = cv2.imread(images_dir + "/" + image_name, cv2.IMREAD_UNCHANGED)

        scene_data = {
                "texts": [text_prompt],
                # "images": np.stack((rgb0, rgb1), axis=0)
                "images": rgb0
            }
        # seg0, seg1 = segmentator(scene_data)
        seg0 = segmentator(scene_data)
        scene_data = {
                "texts": [text_prompt],
                "images": rgb1
            }
        seg1 = segmentator(scene_data)

        if show:
            segmented_bgr0 = bgr0 * seg0[...,None]
            cv2.imshow("Segmentation 0", segmented_bgr0/255)
            segmented_bgr1 = bgr1 * seg1[...,None]
            cv2.imshow("Segmentation 1", segmented_bgr1/255)

        T_C0, T_C1, valid_pair = get_transformations(gray0_tag, bgr0_tag, gray1_tag, bgr1_tag)
        if valid_pair:
            print("SAAAAAVEEEE")
            # saver.save_all_data(rgb0, rgb1, depth0, depth1,
            #                     seg0, seg1, K_cam, T_WC, T_C0, T_C1)

def get_data(id_range, text_prompt):
    text_prompt = 'An image of a ' + text_prompt

    for i in range(0, len(id_range), 2):
        image_name = f"head_camera_rgb_{id_range[i]}.png"
        bgr0_tag = cv2.imread(images_dir + "/" + image_name, )
        gray0_tag = cv2.cvtColor(bgr0_tag, cv2.COLOR_BGR2GRAY)

        image_name = f"head_camera_rgb_{id_range[i+1]}.png"
        bgr0 = cv2.imread(images_dir + "/" + image_name, )
        rgb0 = bgr0.copy()[:,:,[2,1,0]]

        image_name = f"head_camera_depth_to_rgb_{id_range[i+1]}.png"
        depth0 = cv2.imread(images_dir + "/" + image_name, cv2.IMREAD_UNCHANGED)

        scene_data = {
                "texts": [text_prompt],
                "images": rgb0
            }
        seg0 = segmentator(scene_data)

        for j in range(0, len(id_range), 2):
            if i != j:
                print(f"[INFO] Processing the pair {(i/2+1) * (j/2+1)} of {text_prompt}")

                image_name = f"head_camera_rgb_{id_range[j]}.png"
                bgr1_tag = cv2.imread(images_dir + "/" + image_name, )
                gray1_tag = cv2.cvtColor(bgr1_tag, cv2.COLOR_BGR2GRAY)

                image_name = f"head_camera_rgb_{id_range[j+1]}.png"
                bgr1 = cv2.imread(images_dir + "/" + image_name, )
                rgb1 = bgr1.copy()[:,:,[2,1,0]]

                image_name = f"head_camera_depth_to_rgb_{id_range[j+1]}.png"
                depth1 = cv2.imread(images_dir + "/" + image_name, cv2.IMREAD_UNCHANGED)

                scene_data = {
                        "texts": [text_prompt],
                        "images": rgb1
                    }
                seg1 = segmentator(scene_data)

                if show:
                    segmented_bgr0 = bgr0 * seg0[...,None]
                    cv2.imshow("Segmentation 0", segmented_bgr0/255)
                    segmented_bgr1 = bgr1 * seg1[...,None]
                    cv2.imshow("Segmentation 1", segmented_bgr1/255)

                T_C0, T_C1, valid_pair = get_transformations(gray0_tag, bgr0_tag, gray1_tag, bgr1_tag)
                T_delta_C, T_delta_B = get_delta_transformation(T_C0, T_C1)
                if valid_pair:
                    print("SAAAAAVEEEE")
                    # T_W0 = T_WC @ T_C0
                    # T_W1 = T_WC @ T_C1
                    # saver.save_data(rgb0, rgb1, depth0, depth1,
                    #                 seg0, seg1, K_cam, T_WC, T_W0, T_W1,
                    #                 T_delta_C, T_delta_B)



if pose_folder == 1:
    ############# Red toaster
    last_id = 9
    id_range = np.linspace(0, last_id, last_id + 1, dtype=int)
    text_prompt = 'red toaster'
    get_data(id_range, text_prompt)

    ############# Silver toaster
    first_id = last_id+1
    last_id = 19
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    text_prompt = 'silver toaster'
    get_data(id_range, text_prompt)

    ############# White pot
    first_id = last_id+1
    last_id = 29
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    text_prompt = 'white pot'
    get_data(id_range, text_prompt)

    ############# Green box
    first_id = last_id+1
    last_id = 39
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    text_prompt = 'light green plastic'
    get_data(id_range, text_prompt)

    ############# Black box
    first_id = last_id+1
    last_id = 47
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    text_prompt = 'black box'
    get_data(id_range, text_prompt)

    ############# White box
    first_id = last_id+1
    last_id = 53
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    text_prompt = 'white cylinder'
    get_data(id_range, text_prompt)

    ############# Blue pan
    first_id = last_id+1
    last_id = 64
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    id_range = np.delete(id_range, 4) #image 58 wasn't taken
    text_prompt = 'blue pan'
    get_data(id_range, text_prompt)




elif pose_folder == 2:
    ############# Red toaster
    last_id = 9
    id_range = np.linspace(0, last_id, last_id + 1, dtype=int)
    text_prompt = 'red toaster'
    get_data(id_range, text_prompt)
    

    ############# Silver toaster
    print("############### Switching Object ###############")
    first_id = last_id+1
    last_id = 19
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    text_prompt = 'silver toaster'
    get_data(id_range, text_prompt)
    

    ############# Green box
    print("############### Switching Object ###############")
    first_id = last_id+1
    last_id = 29
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    text_prompt = 'light green plastic'
    get_data(id_range, text_prompt)

    ############# Blue pan
    print("############### Switching Object ###############")
    first_id = last_id+1
    last_id = 37
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    # text_prompt = 'blue and brown pan'
    text_prompt = 'beiche pan with handle'
    get_data(id_range, text_prompt)

    ############# Blue pan
    print("############### Switching Object ###############")
    first_id = last_id+2
    last_id = 46
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    text_prompt = 'beiche pan with handle'
    get_data(id_range, text_prompt)

    ############ Wooden box
    print("############### Switching Object ###############")
    first_id = 50
    last_id = 57
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    text_prompt = 'wooden box'
    get_data(id_range, text_prompt)

    ############# White pot
    print("############### Switching Object ###############")
    first_id = last_id+1
    last_id = 65
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    text_prompt = 'white pot with handle'
    get_data(id_range, text_prompt)

    ############# Black box
    print("############### Switching Object ###############")
    first_id = last_id+1
    last_id = 73
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    text_prompt = 'black box'
    get_data(id_range, text_prompt)

    ############# Black box
    print("############### Switching Object ###############")
    first_id = last_id+1
    last_id = 81
    id_range = np.linspace(first_id, last_id, last_id-first_id+1, dtype=int)
    text_prompt = 'gray broom'
    get_data(id_range, text_prompt)






