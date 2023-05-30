# AprilTags
Repository for pose estimation of Apriltags using the apriltag package and OpenCV. This can be used for real world pose estimation ground truth data.

The detect.py script can be used to detect apriltags and find their pose. You can test it with the image in the folder images/
The code in rel_pose_dataset.py can be used to segment out objects and get their pose through the apriltags, as well as collecting segmented rgb and depth.
