import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import click
import cv2
import numpy as np
import sophus as sp
from aria_data_utils.detector_wrappers.april_tag_detector import AprilTagDetectorWrapper
from aria_data_utils.detector_wrappers.object_detector import ObjectDetectorWrapper
from aria_data_utils.image_utils import decorate_img_with_text
from aria_data_utils.perception.april_tag_pose_estimator import AprilTagPoseEstimator
from bosdyn.client.frame_helpers import get_a_tform_b
from fairotag.scene import Scene
from matplotlib import pyplot as plt
from projectaria_tools.core import calibration, data_provider, mps
from scipy.spatial.transform import Rotation as R
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.models.owlvit import OwlVit
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))
AXES_SCALE = 0.9
STREAM1_NAME = "camera-rgb"
STREAM2_NAME = "camera-slam-left"
STREAM3_NAME = "camera-slam-right"
FILTER_DIST = 2.4  # in meters (distance for valid detection)

############## Simple Helper Methods to keep code clean ##############


class AriaReader:
    """
    This class is used to read data from Aria VRS and MPS files
    It can parse through a VRS stream and detect April tags and objects of interest too

    For April tag detection, it uses the AprilTagPoseEstimator class (please refer to AprilTagPoseEstimator.py & AprilTagDetectorWrapper.py)
    For object detection, it uses the Owl-VIT model (please refer to OwlVit.py & ObjectDetectorWrapper.py)

    It also has a few helpers for image rectification, image rotation, image display, etc; and a few helpers to get VRS and MPS file streaming

    Args:
        vrs_file_path (str): Path to VRS file
        mps_file_path (str): Path to MPS file
        verbose (bool, optional): Verbosity flag. Defaults to False.

    """

    def __init__(self, vrs_file_path: str, mps_file_path: str, verbose=False):
        assert vrs_file_path is not None and os.path.exists(
            vrs_file_path
        ), "Incorrect VRS file path"
        assert mps_file_path is not None and os.path.exists(
            mps_file_path
        ), "Incorrect MPS dir path"

        # Verbosity flag for updating images when passed through detectors (this is different from config.VERBOSE)
        self.verbose = verbose

        self.provider = data_provider.create_vrs_data_provider(vrs_file_path)
        assert self.provider is not None, "Cannot open VRS file"

        self.device_calib = self.provider.get_device_calibration()

        # April tag detector object
        self.april_tag_detection_wrapper = AprilTagDetectorWrapper()

        # Object detection object
        self.object_detection_wrapper = ObjectDetectorWrapper()

        # Aria device camera calibration parameters
        self._src_calib_params = None  # type: ignore
        self._dst_calib_params = None  # type: ignore

        # Closed loop trajectory
        closed_loop_trajectory_file = os.path.join(
            mps_file_path, "closed_loop_trajectory.csv"
        )
        self.mps_trajectory = mps.read_closed_loop_trajectory(
            closed_loop_trajectory_file
        )

        # XYZ trajectory for mps
        self.xyz_trajectory = np.empty([len(self.mps_trajectory), 3])

        # Timestamps for mps in seconds
        self.trajectory_s = np.empty([len(self.mps_trajectory)])

        # Different transformations along the trajectory
        self.ariaWorld_T_device_trajectory = []  # type: List[Any]

        # Setup some generic transforms
        self.device_T_cpf = sp.SE3(
            self.device_calib.get_transform_device_cpf().to_matrix()
        )

        # Initialize Trajectory after setting up device transforms
        self.initialize_trajectory()

        # sensor_calib_list = [device_calib.get_sensor_calib(label) for label in stream_names][0]
        # Record VRS timestamps of interest based upon user input during vrs parsing
        self.vrs_idx_of_interest_list = []  # type: List[Any]

    def plot_rgb_and_trajectory(
        self,
        pose_list: List[sp.SE3],
        rgb: np.ndarray,
        traj_data: np.ndarray = None,
        block: bool = True,
    ):
        """
        Plot RGB image with trajectory

        Args:
            marker_pose (sp.SE3): Pose of marker in frame of reference
            device_pose_list (List[sp.SE3]): List of device poses in frame of reference
            rgb (np.ndarray): RGB image
            traj_data (np.ndarray): Trajectory data
        """
        fig = plt.figure(figsize=plt.figaspect(2.0))
        fig.suptitle("A tale of 2 subplots")

        _ = fig.add_subplot(1, 2, 1)
        plt.imshow(rgb)

        scene = Scene()
        for i in range(len(pose_list)):
            scene.add_camera(
                f"device_{i}",
                pose_in_frame=pose_list[i],
                size=AXES_SCALE,
            )

        plt_ax = scene.visualize(fig=fig, should_return=True)
        plt_ax.plot(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2])
        plt.show(block=block)

    def _rotate_img(self, img: np.ndarray, num_of_rotation: int = 3) -> np.ndarray:
        """
        Rotate image in multiples of 90d degrees

        Args:
            img (np.ndarray): Image to be rotated
            k (int, optional): Number of times to rotate by 90 degrees. Defaults to 3.

        Returns:
            np.ndarray: Rotated image
        """
        img = np.ascontiguousarray(
            np.rot90(img, k=num_of_rotation)
        )  # GOD KNOW WHY THIS IS NEEDED -> https://github.com/clovaai/CRAFT-pytorch/issues/84#issuecomment-574683857
        return img

    def _display(self, img: np.ndarray, stream_name: str, wait: int = 1):
        """
        Display image in a cv2 window

        Args:
            img (np.ndarray): Image to be displayed
            stream_name (str): Stream name
            wait (int, optional): Wait time in ms. Defaults to 1 ms
        """
        cv2.imshow(f"Stream - {stream_name}", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(wait)

    def _create_display_window(self, stream_name: str):
        """
        Create a display window for image

        Args:
            stream_name (str): Stream name
        """
        cv2.namedWindow(f"Stream - {stream_name}", cv2.WINDOW_NORMAL)

    def _rectify_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rectify fisheye image based upon camera calibration parameters
        Ensure you have set self._src_calib_param & self._dst_calib_param

        Args:
            image (np.ndarray): Image to be rectified or undistorted

        Returns:
            np.ndarray: Rectified image
        """
        assert self._src_calib_params is not None and self._dst_calib_params is not None
        rectified_image = calibration.distort_by_calibration(
            image, self._dst_calib_params, self._src_calib_params
        )
        return rectified_image

    def get_vrs_timestamp_from_img_idx(
        self, stream_name: str = STREAM1_NAME, idx_of_interest: int = -1
    ) -> int:
        """
        Get VRS timestamp corresponding to VRS image index

        Args:
            stream_name (str, optional): Stream name. Defaults to STREAM1_NAME.
            idx_of_interest (int, optional): Index of interest. Defaults to -1 i.e. the last position of VRS.

        Returns:
            int: Corresponding VRS timestamp in nanoseconds
        """
        stream_id = self.provider.get_stream_id_from_label(stream_name)
        frame_data = self.provider.get_image_data_by_index(stream_id, idx_of_interest)
        return frame_data[1].capture_timestamp_ns

    def parse_camera_stream(
        self,
        stream_name: str,
        detect_qr: bool = False,
        should_display: bool = True,
        detect_objects: bool = False,
        object_labels: List[str] = None,
        iteration_range: Tuple[int, int] = None,
        reverse: bool = False,
        meta_objects: List[str] = ["hand"],
    ) -> Dict[str, Any]:
        """Parse linearly through a camera stream and return a dict of detections
        Detection types supported:
        - April Tag
        - Object Detection

        Args:
            stream_name (str): Stream name
            detect_qr (bool, optional): Boolean to indicate if QR code (Dock ID) should be detected. Defaults to False.
            should_display (bool, optional): Boolean to indicate if image should be displayed. Defaults to True.
            detect_objects (bool, optional): Boolean to indicate if object detection should be performed. Defaults to False.
            object_labels (List[str], optional): List of object labels to be detected. Defaults to None.
            reverse (bool, optional): Boolean to indicate if VRS stream should be parsed in reverse. Defaults to False.
                                      Useful based on the algorithm to detect objects from VRS stream.

        April tag outputs:
        - "tag_image_list" - List of np.ndarrays of images with detections
        - "tag_image_metadata_list" - List of image metadata
        - "tag_device_T_marker_list" - List of Sophus SE3 transforms from Device frame to marker
           CPF is the center frame of Aria with Z pointing out, X pointing up
           and Y pointing left
           Device frame is the base frame of Aria which is aligned with left-slam camera frame

        Object detection outputs:
        - "object_image_list" - List of np.ndarrays of images with detections
        - "object_image_metadata_list" - List of image metadata
        - "object_image_segment" - List of Int signifying which segment the image
            belongs to; smaller number means latter the segment time-wise
        - "object_score_list" - List of Float signifying the detection score
        """
        # Get stream id from stream name
        stream_id = self.provider.get_stream_id_from_label(stream_name)

        # Get device_T_camera i.e. transformation from camera frame of interest to device frame (i.e. left slam camera frame)
        device_T_camera = sp.SE3(
            self.device_calib.get_transform_device_sensor(stream_name).to_matrix()
        )
        assert device_T_camera is not None

        # Setup camera calibration parameters by over-writing self._src_calib_param & self._dst_calib_param
        self._src_calib_params = self.device_calib.get_camera_calib(stream_name)
        self._dst_calib_params = calibration.get_linear_camera_calibration(
            512, 512, 280, stream_name
        )

        outputs: Dict[str, Any] = {}

        # Setup April tag detection by over-writing self._qr_pose_estimator if needed
        if detect_qr:
            focal_lengths = self._dst_calib_params.get_focal_lengths()  # type:ignore
            principal_point = (
                self._dst_calib_params.get_principal_point()  # type: ignore
            )
            self.april_tag_detection_wrapper.enable_detector()
            outputs.update(
                self.april_tag_detection_wrapper._init_april_tag_detector(
                    focal_lengths=focal_lengths, principal_point=principal_point
                )
            )

        # Setup object detection (Owl-ViT) if needed
        if detect_objects:
            self.object_detection_wrapper.enable_detector()
            outputs.update(
                self.object_detection_wrapper._init_object_detector(
                    object_labels + meta_objects, verbose=self.verbose
                )
            )
            self.object_detection_wrapper._core_objects = object_labels
            self.object_detection_wrapper._meta_objects = meta_objects

        if should_display:
            self._create_display_window(stream_name)

        # Logic for iterating through VRS stream
        num_frames = self.provider.get_num_data(stream_id)
        iteration_delta = -1 if reverse else 1
        if iteration_range is None:
            iteration_range = (0, num_frames)

        if reverse:
            start_frame = iteration_range[1] - 1
            end_frame = iteration_range[0]
        else:
            start_frame = iteration_range[0]
            end_frame = iteration_range[1] - 1
        custom_range = range(start_frame, end_frame, iteration_delta)

        # Iterate through VRS stream
        for frame_idx in custom_range:
            # Get image data for frame
            frame_data = self.provider.get_image_data_by_index(stream_id, frame_idx)
            img = frame_data[0].to_numpy_array()
            img_metadata = frame_data[1]

            # Rectify current image frame
            img = self._rectify_image(image=img)

            # Initialize camera_T_marker to None & object_scores to empty dict for current image frame
            camera_T_marker = None
            object_scores = {}

            # Detect QR code in current image frame
            if detect_qr:
                (
                    img,
                    camera_T_marker,
                ) = self.april_tag_detection_wrapper.process_frame(
                    img_frame=img
                )  # type: ignore

            # Rotate current image frame
            img = self._rotate_img(img=img)

            if self.object_detection_wrapper.is_enabled:
                (img, object_scores) = self.object_detection_wrapper.process_frame(
                    img_frame=img
                )

            # If april tag is detected, compute the transformation of marker in cpf frame
            if camera_T_marker is not None:
                device_T_marker = device_T_camera * camera_T_marker
                img, outputs = self.april_tag_detection_wrapper.get_outputs(
                    img_frame=img,
                    outputs=outputs,
                    device_T_marker=device_T_marker,
                    img_metadata=img_metadata,
                )

            # If object is detected, update the outputs
            if object_scores is not {}:
                img, outputs = self.object_detection_wrapper.get_outputs(
                    img_frame=img,
                    outputs=outputs,
                    object_scores=object_scores,
                    img_metadata=img_metadata,
                )

            # Display current image frame
            if should_display:
                self._display(img=img, stream_name=stream_name)

        cv2.destroyAllWindows()
        return outputs

    def initialize_trajectory(self):
        """
        Initialize trajectory data from MPS file for easy access
        """
        # frame(ariaWorld) is same as frame(device) at the start

        for i in range(len(self.mps_trajectory)):
            self.trajectory_s[i] = self.mps_trajectory[
                i
            ].tracking_timestamp.total_seconds()
            ariaWorld_T_device = sp.SE3(
                self.mps_trajectory[i].transform_world_device.to_matrix()
            )
            self.ariaWorld_T_device_trajectory.append(ariaWorld_T_device)
            self.xyz_trajectory[i, :] = ariaWorld_T_device.translation()
            # self.quat_trajectory[i,:] = self.mps_trajectory[i].transform_world_device.quaternion()
        assert len(self.trajectory_s) == len(self.ariaWorld_T_device_trajectory)

    def get_closest_mps_idx_to_timestamp_ns(self, timestamp_ns_of_interest: int) -> int:
        """
        Returns the index of the closest MPS timestamp to the VRS timestamp.
        VRS & MPS timestamps are NOT 100% synced

        Args:
            timestamp_of_interest (int): VRS timestamp in nanoseconds

        Returns:
            mps_idx_of_interest (int): Index of closest MPS timestamp to given VRS timestamp
        """
        # VRS timestamps are NOT 100% synced with MPS timestamps
        # So we find the closest MPS timestamp to the VRS timestamp
        mps_idx_of_interest = np.argmin(
            np.abs(self.trajectory_s * 1e9 - timestamp_ns_of_interest)
        )

        return mps_idx_of_interest

    def get_closest_ariaWorld_T_device_to_timestamp(
        self, timestamp_ns_of_interest: int
    ) -> sp.SE3:
        """
        Returns the transformation of aria device frame to aria world frame at the
        closest MPS timestamp to the given VRS timestamp.
        VRS & MPS timestamps are NOT 100% synced

        Args:
            timestamp_of_interest (int): VRS timestamp in nanoseconds

        Returns:
            sp_transform_of_interest (sp.SE3): Transformation of aria device frame to aria world frame
        """
        mps_idx_of_interest = self.get_closest_mps_idx_to_timestamp_ns(
            timestamp_ns_of_interest
        )
        ariaWorld_T_device_of_insterest = self.ariaWorld_T_device_trajectory[
            mps_idx_of_interest
        ]
        return ariaWorld_T_device_of_insterest

    def get_avg_ariaWorld_T_marker(
        self,
        img_metadata_list: List,
        device_T_marker_list: List,
        filter_dist: float = FILTER_DIST,
    ) -> sp.SE3:
        """
        Returns the average transformation of aria world frame to marker frame

        We get a device_T_marker for each frame in which marker is detected.
        Depending on the frame rate of image capture, multiple frames may have captured the marker.
        Averaging all transforms would be best way to compensate for any noise that may exist in any frame's detections
        camera_T_marker is used to compute device_T_marker[i] and thus ariaWorld_T_marker[i].
        Then we average all ariaWorld_T-marker to find average marker pose wrt ariaWorld.

        NOTE: To compute average of SE3 matrix, we find the average of translation and rotation separately.
              The average rotation is obtained by averaging the quaternions.
        NOTE: Since multiple quaternions can represent the same rotation, we ensure that the 'w' component of the
              quaternion is always positive for effective averaging.

        Args:
            img_metadata_list (List): List of image metadata
            device_T_marker_list (List): List of Sophus SE3 transforms from Device frame to marker
            filter_dist (float, optional): Distance threshold for valid detections. Defaults to FILTER_DIST.
        """
        marker_position_list = []
        marker_quaternion_list = []

        for img_metadata, device_T_marker in zip(
            img_metadata_list, device_T_marker_list
        ):
            vrs_timestamp_of_interest_ns = (
                img_metadata.capture_timestamp_ns
            )  # maybe this can be replaced
            ariaWorld_T_device = self.get_closest_ariaWorld_T_device_to_timestamp(
                vrs_timestamp_of_interest_ns
            )
            ariaWorld_T_marker = ariaWorld_T_device * device_T_marker

            marker_position = ariaWorld_T_marker.translation()
            device_position = ariaWorld_T_device.translation()
            delta = marker_position - device_position
            dist = np.linalg.norm(delta)

            # Consider only those detections where detected marker is within a certain distance of the camera
            if dist < filter_dist:
                marker_position_list.append(marker_position)
                quat = R.from_matrix(ariaWorld_T_marker.rotationMatrix()).as_quat()

                # Ensure quaternion's w is always positive for effective averaging as multiple quaternions can represent the same rotation
                if quat[3] > 0:
                    quat = -1.0 * quat
                marker_quaternion_list.append(quat)

        marker_position_np = np.array(marker_position_list)
        avg_marker_position = np.mean(marker_position_np, axis=0)

        marker_quaternion_np = np.array(marker_quaternion_list)
        avg_marker_quaternion = np.mean(marker_quaternion_np, axis=0)

        avg_ariaWorld_T_marker = sp.SE3(
            R.from_quat(avg_marker_quaternion).as_matrix(), avg_marker_position
        )

        return avg_ariaWorld_T_marker


class SpotQRDetector:
    def __init__(self, spot: Spot):
        self.spot = spot
        print("...Spot initialized...")

    def _to_camera_metadata_dict(self, camera_intrinsics):
        """Converts a camera intrinsics proto to a 3x3 matrix as np.array"""
        intrinsics = {
            "fx": camera_intrinsics.focal_length.x,
            "fy": camera_intrinsics.focal_length.x,
            "ppx": camera_intrinsics.principal_point.x,
            "ppy": camera_intrinsics.principal_point.y,
            "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        return intrinsics

    def _get_body_T_handcam(self, frame_tree_snapshot_hand):
        hand_bd_wrist_T_handcam_dict = (
            frame_tree_snapshot_hand.child_to_parent_edge_map.get(
                "hand_color_image_sensor"
            ).parent_tform_child
        )
        hand_mn_wrist_T_handcam = self.spot.convert_transformation_from_BD_to_magnum(
            hand_bd_wrist_T_handcam_dict
        )

        hand_bd_body_T_wrist_dict = (
            frame_tree_snapshot_hand.child_to_parent_edge_map.get(
                "arm0.link_wr1"
            ).parent_tform_child
        )
        hand_mn_body_T_wrist = self.spot.convert_transformation_from_BD_to_magnum(
            hand_bd_body_T_wrist_dict
        )

        hand_mn_body_T_handcam = hand_mn_body_T_wrist @ hand_mn_wrist_T_handcam

        return hand_mn_body_T_handcam

    def get_spot_a_T_b(self, a: str, b: str) -> sp.SE3:
        frame_tree_snapshot = (
            self.spot.get_robot_state().kinematic_state.transforms_snapshot
        )
        se3_pose = get_a_tform_b(frame_tree_snapshot, a, b)
        pos = se3_pose.get_translation()
        quat = se3_pose.rotation.normalize()
        return sp.SE3(quat.to_matrix(), pos)

    def _get_body_T_headcam(self, frame_tree_snapshot_head):
        head_bd_fr_T_frfe_dict = frame_tree_snapshot_head.child_to_parent_edge_map.get(
            "frontright_fisheye"
        ).parent_tform_child
        head_mn_fr_T_frfe_dict = self.spot.convert_transformation_from_BD_to_magnum(
            head_bd_fr_T_frfe_dict
        )

        head_bd_head_T_fr_dict = frame_tree_snapshot_head.child_to_parent_edge_map.get(
            "frontright"
        ).parent_tform_child
        head_mn_head_T_fr = self.spot.convert_transformation_from_BD_to_magnum(
            head_bd_head_T_fr_dict
        )

        head_bd_body_T_head_dict = (
            frame_tree_snapshot_head.child_to_parent_edge_map.get(
                "head"
            ).parent_tform_child
        )
        head_mn_body_T_head = self.spot.convert_transformation_from_BD_to_magnum(
            head_bd_body_T_head_dict
        )

        head_mn_head_T_frfe = head_mn_head_T_fr @ head_mn_fr_T_frfe_dict
        head_mn_body_T_frfe = head_mn_body_T_head @ head_mn_head_T_frfe

        return head_mn_body_T_frfe

    def _get_spotWorld_T_handcam(
        self, frame_tree_snapshot_hand, spot_frame: str = "vision"
    ):
        if spot_frame != "vision" and spot_frame != "odom":
            raise ValueError("spot_frame should be either vision or odom")
        spot_world_frame = spot_frame

        hand_bd_wrist_T_handcam_dict = (
            frame_tree_snapshot_hand.child_to_parent_edge_map.get(
                "hand_color_image_sensor"
            ).parent_tform_child
        )
        hand_mn_wrist_T_handcam = self.spot.convert_transformation_from_BD_to_magnum(
            hand_bd_wrist_T_handcam_dict
        )

        hand_bd_body_T_wrist_dict = (
            frame_tree_snapshot_hand.child_to_parent_edge_map.get(
                "arm0.link_wr1"
            ).parent_tform_child
        )
        hand_mn_body_T_wrist = self.spot.convert_transformation_from_BD_to_magnum(
            hand_bd_body_T_wrist_dict
        )

        hand_bd_body_T_spotWorld_dict = (
            frame_tree_snapshot_hand.child_to_parent_edge_map.get(
                spot_world_frame
            ).parent_tform_child
        )
        hand_mn_body_T_spotWorld = self.spot.convert_transformation_from_BD_to_magnum(
            hand_bd_body_T_spotWorld_dict
        )
        hand_mn_spotWorld_T_body = hand_mn_body_T_spotWorld.inverted()

        hand_mn_body_T_handcam = hand_mn_body_T_wrist @ hand_mn_wrist_T_handcam
        hand_mn_spotWorld_T_handcam = hand_mn_spotWorld_T_body @ hand_mn_body_T_handcam

        return hand_mn_spotWorld_T_handcam

    def _get_spotWorld_T_headcam(
        self, frame_tree_snapshot_head, use_vision_as_world: bool = True
    ):
        spot_world_frame = "vision" if use_vision_as_world else "odom"

        head_bd_fr_T_frfe_dict = frame_tree_snapshot_head.child_to_parent_edge_map.get(
            "frontright_fisheye"
        ).parent_tform_child
        head_mn_fr_T_frfe_dict = self.spot.convert_transformation_from_BD_to_magnum(
            head_bd_fr_T_frfe_dict
        )

        head_bd_head_T_fr_dict = frame_tree_snapshot_head.child_to_parent_edge_map.get(
            "frontright"
        ).parent_tform_child
        head_mn_head_T_fr = self.spot.convert_transformation_from_BD_to_magnum(
            head_bd_head_T_fr_dict
        )

        head_bd_body_T_head_dict = (
            frame_tree_snapshot_head.child_to_parent_edge_map.get(
                "head"
            ).parent_tform_child
        )
        head_mn_body_T_head = self.spot.convert_transformation_from_BD_to_magnum(
            head_bd_body_T_head_dict
        )

        head_bd_body_T_spotWorld_dict = (
            frame_tree_snapshot_head.child_to_parent_edge_map.get(
                spot_world_frame
            ).parent_tform_child
        )
        head_mn_body_T_spotWorld = self.spot.convert_transformation_from_BD_to_magnum(
            head_bd_body_T_spotWorld_dict
        )
        head_mn_spotWorld_T_body = head_mn_body_T_spotWorld.inverted()

        head_mn_head_T_frfe = head_mn_head_T_fr @ head_mn_fr_T_frfe_dict
        head_mn_body_T_frfe = head_mn_body_T_head @ head_mn_head_T_frfe
        head_mn_spotWorld_T_frfe = head_mn_spotWorld_T_body @ head_mn_body_T_frfe

        return head_mn_spotWorld_T_frfe

    def get_avg_spotWorld_T_marker_HAND(
        self,
        spot_frame: str = "vision",
        data_size_for_avg: int = 10,
        filter_dist: float = 2.2,
    ):
        """
        Returns the average transformation of spot world frame to marker frame

        We get a camera_T_marker for each frame in which marker is detected.
        Depending on the frame rate of image capture, multiple frames may have captured the marker.
        Averaging all transforms would be best way to compensate for any noise that may exist in any frame's detections
        camera_T_marker is used to compute spot_T_marker[i] and thus spotWorld_T_marker[i].
        Then we average all spotWorld_T-marker to find average marker pose wrt spotWorld.
        """
        if spot_frame != "vision" and spot_frame != "odom":
            raise ValueError("base_frame should be either vision or odom")
        cv2.namedWindow("hand_image", cv2.WINDOW_AUTOSIZE)
        spot_world_frame = spot_frame

        # Get Hand camera intrinsics
        hand_cam_intrinsics = self.spot.get_camera_intrinsics(SpotCamIds.HAND_COLOR)
        hand_cam_intrinsics = self._to_camera_metadata_dict(hand_cam_intrinsics)
        hand_cam_pose_estimator = AprilTagPoseEstimator(hand_cam_intrinsics)

        # Register marker ids
        marker_ids_list = [DOCK_ID]
        # marker_ids_list = [i for i in range(521, 550)]
        hand_cam_pose_estimator.register_marker_ids(marker_ids_list)

        marker_position_from_dock_list = []  # type: List[Any]
        marker_quaternion_form_dock_list = []  # type: List[Any]

        marker_position_from_robot_list = []  # type: List[Any]
        marker_quaternion_form_robot_list = []  # type: List[Any]

        while len(marker_position_from_dock_list) < data_size_for_avg:
            print(f"Iterating - {len(marker_position_from_dock_list)}")
            is_marker_detected_from_hand_cam = False
            img_response_hand = self.spot.get_hand_image()
            img_hand = image_response_to_cv2(img_response_hand)

            (
                img_rend_hand,
                hand_mn_handcam_T_marker,
            ) = hand_cam_pose_estimator.detect_markers_and_estimate_pose(
                img_hand, should_render=True
            )

            if hand_mn_handcam_T_marker is not None:
                print("Trackedddd")
                is_marker_detected_from_hand_cam = True

            # Spot - spotWorld_T_handcam computation
            frame_tree_snapshot_hand = img_response_hand.shot.transforms_snapshot
            hand_mn_body_T_handcam = self._get_body_T_handcam(frame_tree_snapshot_hand)
            hand_mn_spotWorld_T_handcam = self._get_spotWorld_T_handcam(
                frame_tree_snapshot_hand, spot_frame=spot_frame
            )

            if is_marker_detected_from_hand_cam:
                hand_mn_spotWorld_T_marker = (
                    hand_mn_spotWorld_T_handcam @ hand_mn_handcam_T_marker
                )

                hand_mn_body_T_marker = (
                    hand_mn_body_T_handcam @ hand_mn_handcam_T_marker
                )

                img_rend_hand = decorate_img_with_text(
                    img=img_rend_hand,
                    frame_name=spot_world_frame,
                    position=hand_mn_spotWorld_T_marker.translation,
                )

                dist = hand_mn_handcam_T_marker.translation.length()

                print(
                    f"Dist = {dist}, Recordings - {len(marker_position_from_dock_list)}"
                )
                if dist < filter_dist:
                    marker_position_from_dock_list.append(
                        np.array(hand_mn_spotWorld_T_marker.translation)
                    )
                    marker_quaternion_form_dock_list.append(
                        R.from_matrix(hand_mn_spotWorld_T_marker.rotation()).as_quat()
                    )
                    marker_position_from_robot_list.append(
                        np.array(hand_mn_body_T_marker.translation)
                    )
                    marker_quaternion_form_robot_list.append(
                        R.from_matrix(hand_mn_body_T_marker.rotation()).as_quat()
                    )

            cv2.imshow("hand_image", img_rend_hand)
            cv2.waitKey(1)

        marker_position_from_dock_np = np.array(marker_position_from_dock_list)
        avg_marker_position_from_dock = np.mean(marker_position_from_dock_np, axis=0)

        marker_quaternion_from_dock_np = np.array(marker_quaternion_form_dock_list)
        avg_marker_quaternion_from_dock = np.mean(
            marker_quaternion_from_dock_np, axis=0
        )

        marker_position_from_robot_np = np.array(marker_position_from_robot_list)
        avg_marker_position_from_robot = np.mean(marker_position_from_robot_np, axis=0)

        marker_quaternion_from_robot_np = np.array(marker_quaternion_form_robot_list)
        avg_marker_quaternion_from_robot = np.mean(
            marker_quaternion_from_robot_np, axis=0
        )

        avg_spotWorld_T_marker = sp.SE3(
            R.from_quat(avg_marker_quaternion_from_dock).as_matrix(),
            avg_marker_position_from_dock,
        )
        avg_spot_T_marker = sp.SE3(
            R.from_quat(avg_marker_quaternion_from_robot).as_matrix(),
            avg_marker_position_from_robot,
        )

        return avg_spotWorld_T_marker, avg_spot_T_marker

    def get_avg_spotWorld_T_marker_HEAD(
        self,
        use_vision_as_world: bool = True,
        data_size_for_avg: int = 10,
        filter_dist: float = FILTER_DIST,
    ):
        pass

    def get_avg_spotWorld_T_marker(
        self,
        camera: str = "hand",
        use_vision_as_world: bool = True,
        data_size_for_avg: int = 10,
        filter_dist: float = FILTER_DIST,
    ):
        pass


@click.command()
@click.option("--data-path", help="Path to the data directory", type=str)
@click.option("--vrs-name", help="Name of the vrs file", type=str)
@click.option("--dry-run", type=bool, default=False)
@click.option("--verbose", type=bool, default=True)
@click.option("--use-spot/--no-spot", default=True)
@click.option("--object-names", type=str, multiple=True, default=["smartphone"])
@click.option("--qr/--no-qr", default=True)
def main(
    data_path: str,
    vrs_name: str,
    dry_run: bool,
    verbose: bool,
    use_spot: bool,
    object_names: List[str],
    qr: bool,
):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("VRS_MPS_STREAMER")
    vrsfile = os.path.join(data_path, vrs_name + ".vrs")
    vrs_mps_streamer = AriaReader(
        vrs_file_path=vrsfile, mps_file_path=data_path, verbose=verbose
    )

    outputs = vrs_mps_streamer.parse_camera_stream(
        stream_name=STREAM1_NAME,
        detect_qr=qr,
        detect_objects=True,
        object_labels=list(object_names),
        reverse=True,
    )
    # tag_img_list = outputs["tag_image_list"]
    if qr:
        tag_img_metadata_list = outputs["tag_image_metadata_list"]
        tag_device_T_marker_list = outputs["tag_device_T_marker_list"]

        avg_ariaWorld_T_marker = vrs_mps_streamer.get_avg_ariaWorld_T_marker(
            tag_img_metadata_list,
            tag_device_T_marker_list,
            filter_dist=FILTER_DIST,
        )
        logger.debug(avg_ariaWorld_T_marker)
        avg_marker_T_ariaWorld = avg_ariaWorld_T_marker.inverse()
    else:
        avg_ariaWorld_T_marker = sp.SE3()

    # find best device location, wrt AriaWorld, for best scored object detection
    # FIXME: extend to multiple objects

    # object detection metadata
    best_object_frame_idx = {}
    best_object_frame_timestamp_ns = {}
    best_object_ariaWorld_T_device = {}
    best_object_img = {}
    for object_name in object_names:
        # TODO: what happens when object is not detected?
        best_object_frame_idx[object_name] = outputs["object_score_list"][
            object_name
        ].index(max(outputs["object_score_list"][object_name]))
        best_object_frame_timestamp_ns[object_name] = outputs[
            "object_image_metadata_list"
        ][object_name][best_object_frame_idx[object_name]].capture_timestamp_ns
        best_object_ariaWorld_T_device[object_name] = (
            vrs_mps_streamer.get_closest_ariaWorld_T_device_to_timestamp(
                best_object_frame_timestamp_ns[object_name]
            )
        )
        best_object_img[object_name] = outputs["object_image_list"][object_name][
            best_object_frame_idx[object_name]
        ]

    if use_spot:
        spot = Spot("ArmKeyboardTeleop")
        spot_qr = SpotQRDetector(spot=spot)
        (
            avg_spotWorld_T_marker,
            avg_spot_T_marker,
        ) = spot_qr.get_avg_spotWorld_T_marker_HAND()

        logger.debug(avg_spotWorld_T_marker)

        avg_spotWorld_T_ariaWorld = avg_spotWorld_T_marker * avg_marker_T_ariaWorld
        avg_ariaWorld_T_spotWorld = avg_spotWorld_T_ariaWorld.inverse()

        avg_spot_T_ariaWorld = avg_spot_T_marker * avg_marker_T_ariaWorld
        avg_ariaWorld_T_spot = avg_spot_T_ariaWorld.inverse()

        spotWorld_T_device_trajectory = np.array(
            [
                (avg_spotWorld_T_ariaWorld * ariaWorld_T_device).translation()
                for ariaWorld_T_device in vrs_mps_streamer.ariaWorld_T_device_trajectory
            ]
        )

    for i in range(len(object_names)):
        # choose the next object to pick
        next_object = object_names[i]

        next_object_ariaWorld_T_device = best_object_ariaWorld_T_device[next_object]
        next_object_ariaWorld_T_cpf = (
            next_object_ariaWorld_T_device * vrs_mps_streamer.device_T_cpf
        )
        if use_spot:
            # get the best object pose in spotWorld frame
            next_object_spotWorld_T_cpf = (
                avg_spotWorld_T_ariaWorld
                * next_object_ariaWorld_T_device
                * vrs_mps_streamer.device_T_cpf
            )
            vrs_mps_streamer.plot_rgb_and_trajectory(
                pose_list=[
                    avg_spotWorld_T_marker,
                    avg_spotWorld_T_ariaWorld * vrs_mps_streamer.device_T_cpf,
                    spot_qr.get_spot_a_T_b("vision", "body"),
                    next_object_spotWorld_T_cpf,
                ],
                rgb=np.zeros((10, 10, 3), dtype=np.uint8),
                traj_data=spotWorld_T_device_trajectory,
                block=False,
            )
            vrs_mps_streamer.plot_rgb_and_trajectory(
                pose_list=[
                    avg_ariaWorld_T_marker,
                    avg_ariaWorld_T_spotWorld,
                    avg_ariaWorld_T_spot,
                    next_object_spotWorld_T_cpf,
                ],
                rgb=np.zeros((10, 10, 3), dtype=np.uint8),
                traj_data=vrs_mps_streamer.xyz_trajectory,
                block=True,
            )
            pose_of_interest = next_object_spotWorld_T_cpf
            position = pose_of_interest.translation()
            logger.debug(f" Going to {next_object=} at {position=}")
            if not dry_run:
                skill_manager = SpotSkillManager()
                skill_manager.nav(
                    position[0], position[1], 0.0
                )  # FIXME: get yaw from Aria Pose
                skill_manager.pick(next_object)
        else:
            logger.debug(f"Showing {next_object=}")
            vrs_mps_streamer.plot_rgb_and_trajectory(
                pose_list=[
                    avg_ariaWorld_T_marker,
                    next_object_ariaWorld_T_cpf,
                ],
                rgb=(
                    best_object_img[next_object]
                    if best_object_img
                    else np.zeros((10, 10, 3), dtype=np.uint8)
                ),
                traj_data=vrs_mps_streamer.xyz_trajectory,
                block=True,
            )


if __name__ == "__main__":
    main()
