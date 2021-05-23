import argparse
import threading
import time
from pathlib import Path
from pose import getKeypoints, getValidPairs, getPersonwiseKeypoints
import cv2
import depthai as dai
import numpy as np
from imutils.video import FPS
import math
import matplotlib.pyplot as plt
import csv
import os



################################################################################
# get arguments
################################################################################
parser = argparse.ArgumentParser()

parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-ccam', '--ccamera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid and -mcam)")
parser.add_argument('-mcam', '--mcamera', action="store_true", help="Use DepthAI stereo cameras for inference (conflicts with -vid and -ccam)")
parser.add_argument('-vid1', '--video1', type=str, help="Path to 1st video file to be used for inference (conflicts with -cam)")
parser.add_argument('-vid2', '--video2', type=str, help="Path to 2nd video file to be used for inference (conflicts with -cam)")
parser.add_argument("-depth", "--depth", help="enables depth calculations with monocameras",default=False, action="store_true")


args = parser.parse_args()

if not args.ccamera and not args.video1 and not args.mcamera:
    raise RuntimeError("No source selected. Please use either \"-ccam\" to use RGB camera as a source or \"-mcam\" to use mono cameras with depth as a source or \"-vid <path>\" to run on video")

debug = not args.no_debug
depth_bool = args.depth

################################################################################
# function definitions
################################################################################
"""
flattening opencv arrays
"""
def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()


"""
gets angle based on 3 points where b is the center point between a and c.
assumes segments are drawn between ab and bc
"""
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang

"""
Plotting kinematic and kinetic data with norms and foot-offs
"""
def pplot(time, subplot_pos, left_norm, right_norm, title, xlabel, ylabel,
dict, dictkey):
# pplot(time, subplot_pos, left_norm, right_norm, title, xlabel, ylabel,
# ylim_min, ylim_max, lfo_norm, rfo_norm, dict, dictkey):
    if dict != None and dictkey != None:
        vals = dict.get(dictkey)
        tn2 = np.linspace(0, 100, len(vals[0]))
        y2 = np.interp(time, tn2, vals[0])
        ed = np.interp(time, tn2, vals[1])
    plt.subplot(2, 1, subplot_pos)  # Layout 1 rows, 2 columns, position
    plt.subplots_adjust(left=None, bottom=None,
    right=None, top=None, wspace=0.5, hspace=0.5)

    left_masked = np.ma.masked_where(left_norm < -360, left_norm)
    right_masked = np.ma.masked_where(right_norm < -360, right_norm)
    left_masked = np.ma.masked_where(left_masked > 360, left_masked)
    right_masked = np.ma.masked_where(right_masked > 360, right_masked)

    plot1 = plt.plot(time, left_masked, '#DC143C', time, right_masked, '#14C108')
    plt.title(title,fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim(ylim_min, ylim_max)
    plt.axhline(0, color='black')
    # plt.axvline(lfo_norm,color='#DC143C',linewidth=1)
    # plt.axvline(rfo_norm,color='#14C108',linewidth=1)
    if dict !=None:
        plt.errorbar(time,y2,ed,linestyle='None',marker=',',mfc="#B8B4B4",
        mec="#B8B4B4",c="#B8B4B4",dash_capstyle='round')
    plt.grid(color='#A2A2A2',which='major',axis='y',linestyle='-',linewidth='1')
    plt.tick_params(axis='x',which='major',direction='in',
    color='#A2A2A2',length=10)

    return plot1

"""
Old pipeline creation for color camera and video file input from
depthai-experiments/gen2-human-pose
"""
def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()
    print("Creating Human Pose Estimation Neural Network...")
    if args.ccamera or args.video1:
        # NeuralNetwork
        pose_nn = pipeline.createNeuralNetwork()
        if args.ccamera:
            pose_nn.setBlobPath(str(Path("models/human-pose-estimation-0001_openvino_2021.2_6shave.blob").resolve().absolute()))
        else:
            pose_nn.setBlobPath(str(Path("models/human-pose-estimation-0001_openvino_2021.2_8shave.blob").resolve().absolute()))
    elif args.mcamera:
        pose_nn_right = pipeline.createNeuralNetwork()
        pose_nn_right.setBlobPath(str(Path("models/human-pose-estimation-0001_openvino_2021.2_6shave.blob").resolve().absolute()))

    if args.ccamera:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam_xout = pipeline.createXLinkOut()

        cam.setPreviewSize(456, 256)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)

        cam_xout.setStreamName("cam_out")

        cam.preview.link(cam_xout.input)
        controlIn = pipeline.createXLinkIn()
        controlIn.setStreamName('control')
        controlIn.out.link(cam.inputControl)

        # Increase threads for detection
        pose_nn.setNumInferenceThreads(2)
        # Specify that network takes latest arriving frame in non-blocking manner
        pose_nn.input.setQueueSize(1)
        pose_nn.input.setBlocking(False)
        pose_nn_xout = pipeline.createXLinkOut()
        pose_nn_xout.setStreamName("pose_nn")
        pose_nn.out.link(pose_nn_xout.input)

        cam.preview.link(pose_nn.input)

    elif args.video1:
        # Increase threads for detection
        pose_nn.setNumInferenceThreads(2)
        # Specify that network takes latest arriving frame in blocking manner
        pose_nn.input.setQueueSize(1)
        pose_nn.input.setBlocking(True)
        pose_nn_xout = pipeline.createXLinkOut()
        pose_nn_xout.setStreamName("pose_nn")
        pose_nn.out.link(pose_nn_xout.input)

        pose_in = pipeline.createXLinkIn()
        pose_in.setStreamName("pose_in")
        pose_in.out.link(pose_nn.input)

        # depth1 = pipeline.createStereoDepth()
        # spatialLocationCalculator1 = pipeline.createSpatialLocationCalculator()
        #
        # xoutDepth1 = pipeline.createXLinkOut()
        # xoutSpatialData1 = pipeline.createXLinkOut()
        # xinSpatialCalcConfig1 = pipeline.createXLinkIn()
        #
        # xoutDepth1.setStreamName("depth1")
        # xoutSpatialData1.setStreamName("spatialData1")
        # xinSpatialCalcConfig1.setStreamName("spatialCalcConfig1")
        #
        # outputDepth = True
        # outputRectified = False
        # lrcheck = False
        # subpixel = False
        #
        #
        # depth1.setOutputDepth(outputDepth)
        # depth1.setOutputRectified(outputRectified)
        # depth1.setConfidenceThreshold(255)
        # depth1.setLeftRightCheck(lrcheck)
        # depth1.setSubpixel(subpixel)

        # cam_left.out.link(depth1.left)
        # cam_right.out.link(depth1.right)
    elif args.mcamera:
        print("Creating Mono Cameras...")
        cam_right = pipeline.createMonoCamera()
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)


        cam_left = pipeline.createMonoCamera()
        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

        if depth_bool:
            # stereo depth
            cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

            #create a node to produce depth map using disparity output
            depth1 = pipeline.createStereoDepth()
            depth2 = pipeline.createStereoDepth()
            spatialLocationCalculator1 = pipeline.createSpatialLocationCalculator()
            spatialLocationCalculator2 = pipeline.createSpatialLocationCalculator()

            xoutDepth1 = pipeline.createXLinkOut()
            xoutDepth2 = pipeline.createXLinkOut()
            xoutSpatialData1 = pipeline.createXLinkOut()
            xinSpatialCalcConfig1 = pipeline.createXLinkIn()
            xoutSpatialData2 = pipeline.createXLinkOut()
            xinSpatialCalcConfig2 = pipeline.createXLinkIn()

            xoutDepth1.setStreamName("depth1")
            xoutDepth2.setStreamName("depth2")
            xoutSpatialData1.setStreamName("spatialData1")
            xinSpatialCalcConfig1.setStreamName("spatialCalcConfig1")
            xoutSpatialData2.setStreamName("spatialData2")
            xinSpatialCalcConfig2.setStreamName("spatialCalcConfig2")

            outputDepth = True
            outputRectified = False
            lrcheck = False
            subpixel = False


            depth1.setOutputDepth(outputDepth)
            depth1.setOutputRectified(outputRectified)
            depth1.setConfidenceThreshold(255)
            depth1.setLeftRightCheck(lrcheck)
            depth1.setSubpixel(subpixel)

            cam_left.out.link(depth1.left)
            cam_right.out.link(depth1.right)

            # spatialLocationCalculator1.passthroughDepth.link(xoutDepth1.input)
            # depth1.depth.link(spatialLocationCalculator1.inputDepth)

            topLeft1 = dai.Point2f(0.2, 0.8)
            bottomRight1 = dai.Point2f(0.3, 0.9)

            spatialLocationCalculator1.setWaitForConfigInput(False)
            config1 = dai.SpatialLocationCalculatorConfigData()
            config1.depthThresholds.lowerThreshold = 100
            config1.depthThresholds.upperThreshold = 10000
            config1.roi = dai.Rect(topLeft1, bottomRight1)
            spatialLocationCalculator1.initialConfig.addROI(config1)
            spatialLocationCalculator1.out.link(xoutSpatialData1.input)
            xinSpatialCalcConfig1.out.link(spatialLocationCalculator1.inputConfig)

            spatialLocationCalculator1.passthroughDepth.link(xoutDepth1.input)
            depth1.depth.link(spatialLocationCalculator1.inputDepth)


            depth2.setOutputDepth(outputDepth)
            depth2.setOutputRectified(outputRectified)
            depth2.setConfidenceThreshold(255)
            depth2.setLeftRightCheck(lrcheck)
            depth2.setSubpixel(subpixel)

            cam_left.out.link(depth2.left)
            cam_right.out.link(depth2.right)

            topLeft2 = dai.Point2f(0.7, 0.8)
            bottomRight2 = dai.Point2f(0.8, 0.9)

            spatialLocationCalculator2.setWaitForConfigInput(False)
            config2 = dai.SpatialLocationCalculatorConfigData()
            config2.depthThresholds.lowerThreshold = 100
            config2.depthThresholds.upperThreshold = 10000
            config2.roi = dai.Rect(topLeft2, bottomRight2)
            spatialLocationCalculator2.initialConfig.addROI(config2)
            spatialLocationCalculator2.out.link(xoutSpatialData2.input)
            xinSpatialCalcConfig2.out.link(spatialLocationCalculator2.inputConfig)

            spatialLocationCalculator2.passthroughDepth.link(xoutDepth2.input)
            depth2.depth.link(spatialLocationCalculator2.inputDepth)

            # Create a node to convert the grayscale frame into the nn-acceptable form
            manip_right = pipeline.createImageManip()
            manip_right.initialConfig.setResize(456, 256)
            # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
            manip_right.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
            cam_right.out.link(manip_right.inputImage)
            manip_right.out.link(pose_nn_right.input)

            controlIn_right = pipeline.createXLinkIn()
            controlIn_right.setStreamName('control_right')
            controlIn_right.out.link(cam_right.inputControl)

            # # Create disparity output
            # xout_manip_disparity = pipeline.createXLinkOut()
            # xout_manip_disparity.setStreamName("disparity")
            # depth.disparity.link(xout_manip_disparity.input)

            # Create outputs
            xout_manip_right = pipeline.createXLinkOut()
            xout_manip_right.setStreamName("right")
            manip_right.out.link(xout_manip_right.input)

            # Increase threads for detection
            pose_nn_right.setNumInferenceThreads(2)
            # Specify that network takes latest arriving frame in non-blocking manner
            pose_nn_right.input.setQueueSize(1)
            pose_nn_right.input.setBlocking(False)
            xout_nn_right = pipeline.createXLinkOut()
            xout_nn_right.setStreamName("pose_nn_right")
            pose_nn_right.out.link(xout_nn_right.input)

        else:
            # discrepancy only
            depth = pipeline.createStereoDepth()
            cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

            # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
            # For depth filtering
            median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

            depth.setConfidenceThreshold(200)
            depth.setLeftRightCheck(False)
            depth.setExtendedDisparity(False)
            cam_left.out.link(depth.left)
            cam_right.out.link(depth.right)

            # Create a node to convert the grayscale frame into the nn-acceptable form
            manip_right = pipeline.createImageManip()
            manip_right.initialConfig.setResize(456, 256)
            # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
            manip_right.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
            cam_right.out.link(manip_right.inputImage)
            manip_right.out.link(pose_nn_right.input)

            controlIn_right = pipeline.createXLinkIn()
            controlIn_right.setStreamName('control_right')
            controlIn_right.out.link(cam_right.inputControl)

            # Create disparity output
            xout_manip_disparity = pipeline.createXLinkOut()
            xout_manip_disparity.setStreamName("disparity")
            depth.disparity.link(xout_manip_disparity.input)

            # Create outputs
            xout_manip_right = pipeline.createXLinkOut()
            xout_manip_right.setStreamName("right")
            manip_right.out.link(xout_manip_right.input)

            # Increase threads for detection
            pose_nn_right.setNumInferenceThreads(2)
            # Specify that network takes latest arriving frame in non-blocking manner
            pose_nn_right.input.setQueueSize(1)
            pose_nn_right.input.setBlocking(False)
            xout_nn_right = pipeline.createXLinkOut()
            xout_nn_right.setStreamName("pose_nn_right")
            pose_nn_right.out.link(xout_nn_right.input)


    print("Pipeline created.")
    return pipeline

"""
Pose function definitions from pose.py
"""
def pose_thread1(in_queue):
    global keypoints_list, detected_keypoints, personwiseKeypoints
    while running:
        try:
            raw_in = in_queue.get()
        except RuntimeError:
            return
        fps.tick('nn')
        heatmaps = np.array(raw_in.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
        pafs = np.array(raw_in.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
        heatmaps = heatmaps.astype('float32')
        pafs = pafs.astype('float32')
        outputs = np.concatenate((heatmaps, pafs), axis=1)

        new_keypoints = []
        new_keypoints_list = np.zeros((0, 3))
        keypoint_id = 0

        for row in range(18):
            probMap = outputs[0, row, :, :]
            probMap = cv2.resize(probMap, (w, h))  # (456, 256)
            keypoints = getKeypoints(probMap, 0.3)
            new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
            keypoints_with_id = []

            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoint_id += 1

            new_keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = getValidPairs(outputs, w, h, new_keypoints)
        newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)

        detected_keypoints, keypoints_list, personwiseKeypoints = (new_keypoints, new_keypoints_list, newPersonwiseKeypoints)



"""
Pose function definitions from pose.py with hardcoded width and height
"""
def pose_thread2(in_queue):
    global keypoints_list, detected_keypoints, personwiseKeypoints
    w=456
    h=256
    while running:
        try:
            raw_in = in_queue.get()
        except RuntimeError:
            return
        fps.tick('nn')
        heatmaps = np.array(raw_in.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
        pafs = np.array(raw_in.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
        heatmaps = heatmaps.astype('float32')
        pafs = pafs.astype('float32')
        outputs = np.concatenate((heatmaps, pafs), axis=1)

        new_keypoints = []
        new_keypoints_list = np.zeros((0, 3))
        keypoint_id = 0

        for row in range(18):
            probMap = outputs[0, row, :, :]
            probMap = cv2.resize(probMap, (w, h))  # (456, 256)
            keypoints = getKeypoints(probMap, 0.3)
            new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
            keypoints_with_id = []

            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoint_id += 1

            new_keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = getValidPairs(outputs, w, h, new_keypoints)
        newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)

        detected_keypoints, keypoints_list, personwiseKeypoints = (new_keypoints, new_keypoints_list, newPersonwiseKeypoints)


"""
FPS counter class
"""
class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None

        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

    def next_iter(self):
        if args.video1:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.time()
            if delay > 0:
                print("frame zero")
                time.sleep(delay)
        self.timestamp = time.time()
        self.frame_cnt += 1

    def tick(self, name):
        if name in self.ticks:
            self.ticks_cnt[name] += 1
        else:
            self.ticks[name] = time.time()
            self.ticks_cnt[name] = 0

    def tick_fps(self, name):
        if name in self.ticks:
            return self.ticks_cnt[name] / (time.time() - self.ticks[name])
        else:
            return 0

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


# if args.ccamera or args.mcamera:
#     fps = FPSHandler()
# else:
#     cap = cv2.VideoCapture(str(Path(args.video1).resolve().absolute()))
#     fps = FPSHandler(cap)

keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

running = True
pose = None
keypoints_list = None
detected_keypoints = None
personwiseKeypoints = None


# original pipeline for human pose from
# depthai-experiments-master/gen2-human-pose
if args.ccamera or args.video1:

    if args.ccamera:
        fps = FPSHandler()
    elif args.video1 and args.video2:
        cap = cv2.VideoCapture(str(Path(args.video1).resolve().absolute()))
        fps = FPSHandler(cap)
        cap2 = cv2.VideoCapture(str(Path(args.video2).resolve().absolute()))
        fps2 = FPSHandler(cap2)
        print("2 videos")
    else:
        cap = cv2.VideoCapture(str(Path(args.video1).resolve().absolute()))
        fps = FPSHandler(cap)

    with dai.Device(create_pipeline()) as device:
        print("Starting pipeline...")
        device.startPipeline()
        if args.ccamera:
            cam_out = device.getOutputQueue("cam_out", 1, True)
            controlQueue = device.getInputQueue('control')
        else:
            pose_in = device.getInputQueue("pose_in")
        pose_nn = device.getOutputQueue("pose_nn", 1, True)
        t = threading.Thread(target=pose_thread1, args=(pose_nn, ))
        t.start()

        def should_run1():
            return cap.isOpened() if args.video1 else True

        def should_run2():
            return cap.isOpened() if args.video2 else True


        def get_frame():
            if args.video1:
                print("read")
                return cap.read()
            else:
                return True, np.array(cam_out.get().getData()).reshape((3, 256, 456)).transpose(1, 2, 0).astype(np.uint8)

        angle_dict1={}
        eyes_list=[]
        lkneeflex_list=[]
        rkneeflex_list=[]
        lhipflex_list=[]
        rhipflex_list=[]

        try:
            count = 0
            frame_count = 0
            frame_stop = None
            location_dict1 = {}
            while should_run1():
                read_correctly, frame = get_frame()
                #read_correctly,frame=cap.read()
                # if args.video2:
                #     read_correctly2,frame2=cap2.read()
                frame_count += 1

                # restarting video so that the rest of the original video NN
                # can finish running
                if frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("restarting video1")
                    frame_count = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    read_correctly, frame = get_frame()
                    # if args.video2:
                    #     print("restarting video2")
                    #     cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    #     read_correctly2,frame2=cap2.read()

                # stop the replayed video when it has done the
                # amount of frames it skipped at beginning
                if frame_stop is not None:
                    if frame_count==frame_stop:
                        break

                h, w = frame.shape[:2]  # 256, 456
                debug_frame = frame.copy()
                # if args.video2:
                #     debug_frame2 = frame2.copy()

                # if args.video1:
                nn_data = dai.NNData()
                nn_data.setLayer("input", to_planar(debug_frame, (456, 256)))
                pose_in.send(nn_data)


                if debug:
                    pos_dict={}
                    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
                        if frame_stop is None:
                            frame_stop = frame_count
                        for i in range(18):
                            for j in range(len(detected_keypoints[i])):
                                cv2.circle(debug_frame, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                                dict = {keypointsMapping[i]: detected_keypoints[i][j][0:2]}
                                pos_dict.update(dict)

                        for i in range(0,len(keypointsMapping)):
                            if keypointsMapping[i] in pos_dict.keys():
                                current_pos = pos_dict.get(keypointsMapping[i])

                                if keypointsMapping[i] in location_dict1.keys():
                                    previous_pos = location_dict1.pop(keypointsMapping[i])
                                    previous_posx = previous_pos[0]
                                    previous_posy = previous_pos[1]

                                    if isinstance(previous_posx,list):
                                        previous_posx.append(current_pos[0])
                                        previous_posy.append(current_pos[1])
                                    else:
                                        previous_posx = [previous_posx]
                                        previous_posy = [previous_posy]
                                        previous_posx.append(current_pos[0])
                                        previous_posy.append(current_pos[1])

                                    dict = {keypointsMapping[i]: [previous_posx, previous_posy]}

                                else:
                                    dict = {keypointsMapping[i]: [current_pos[0], current_pos[1]]}
                            else:
                                if keypointsMapping[i] in location_dict1.keys():
                                    previous_pos = location_dict1.pop(keypointsMapping[i])
                                    previous_posx = previous_pos[0]
                                    previous_posy = previous_pos[1]

                                    if isinstance(previous_posx,list):
                                        previous_posx.append(-1)
                                        previous_posy.append(-1)
                                    else:
                                        previous_posx = [previous_posx]
                                        previous_posy = [previous_posy]
                                        previous_posx.append(-1)
                                        previous_posy.append(-1)

                                    dict = {keypointsMapping[i]: [previous_posx, previous_posy]}

                                else:
                                    dict = {keypointsMapping[i]: [-1, -1]}
                            location_dict1.update(dict)


                        if 'Nose' in pos_dict.keys() and 'R-Eye' in pos_dict.keys() and 'L-Eye' in pos_dict.keys():
                            angle = getAngle(pos_dict.get('L-Eye'),pos_dict.get('Nose'),pos_dict.get('R-Eye'))
                            eyes_list.append(angle)
                            dict = {'EyesAvg': np.mean(eyes_list)}
                            angle_dict1.update(dict)
                            dict = {'Eyes': eyes_list}
                            angle_dict1.update(dict)
                            print("Eyes Angle", angle)
                        else:
                            eyes_list.append(-1000)
                            angle_dict1.update({'Eyes':eyes_list})

                        if 'L-Hip' in pos_dict.keys() and 'L-Knee' in pos_dict.keys() and 'L-Ank' in pos_dict.keys():
                            angle = getAngle(pos_dict.get('L-Hip'),pos_dict.get('L-Knee'),pos_dict.get('L-Ank'))
                            lkneeflex_list.append(angle)
                            dict = {'LKneeFlexAvg': np.mean(lkneeflex_list)}
                            angle_dict1.update(dict)
                            dict = {'LKneeFlex': lkneeflex_list}
                            angle_dict1.update(dict)
                            print("Left Knee Flexion", angle)
                        else:
                            lkneeflex_list.append(-1000)
                            angle_dict1.update({'LKneeFlex':lkneeflex_list})

                        if 'R-Hip' in pos_dict.keys() and 'R-Knee' in pos_dict.keys() and 'R-Ank' in pos_dict.keys():
                            angle = getAngle(pos_dict.get('R-Hip'),pos_dict.get('R-Knee'),pos_dict.get('R-Ank'))
                            rkneeflex_list.append(angle)
                            dict = {'RKneeFlexAvg': np.mean(rkneeflex_list)}
                            angle_dict1.update(dict)
                            dict = {'RKneeFlex': rkneeflex_list}
                            angle_dict1.update(dict)
                            print("Right Knee Flexion", angle)
                        else:
                            rkneeflex_list.append(-1000)
                            angle_dict1.update({'RKneeFlex':rkneeflex_list})

                        if  'L-Sho' in pos_dict.keys() and 'L-Hip' in pos_dict.keys() and 'L-Knee' in pos_dict.keys():
                            angle = getAngle(pos_dict.get('L-Sho'), pos_dict.get('L-Hip'),pos_dict.get('L-Knee'))
                            lhipflex_list.append(angle)
                            dict = {'LHipFlexAvg': np.mean(lhipflex_list)}
                            angle_dict1.update(dict)
                            dict = {'LHipFlex': lhipflex_list}
                            angle_dict1.update(dict)
                            print("Left Hip Flexion", angle)
                        else:
                            lhipflex_list.append(-1000)
                            angle_dict1.update({'LHipFlex':lhipflex_list})

                        if  'R-Sho' in pos_dict.keys() and 'R-Hip' in pos_dict.keys() and 'R-Knee' in pos_dict.keys():
                            angle = getAngle(pos_dict.get('R-Sho'), pos_dict.get('R-Hip'),pos_dict.get('R-Knee'))
                            rhipflex_list.append(angle)
                            dict = {'RHipFlexAvg': np.mean(rhipflex_list)}
                            angle_dict1.update(dict)
                            dict = {'RHipFlex': rhipflex_list}
                            angle_dict1.update(dict)
                            print("Right Hip Flexion", angle)
                        else:
                            rhipflex_list.append(-1000)
                            angle_dict1.update({'RHipFlex':rhipflex_list})


                        for i in range(17):
                            for n in range(len(personwiseKeypoints)):
                                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                                if -1 in index:
                                    continue
                                B = np.int32(keypoints_list[index.astype(int), 0])
                                A = np.int32(keypoints_list[index.astype(int), 1])
                                cv2.line(debug_frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

                        fps.next_iter()
                        if args.video2:
                            fps2.next_iter()

                        cv2.putText(debug_frame, f"MONO FPS: {round(fps.fps(), 1)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                        cv2.putText(debug_frame, f"NN FPS:  {round(fps.tick_fps('nn'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                        cv2.imshow("mono", debug_frame)
                        # if args.video2:
                        #     cv2.imshow("mono2", frame2)

                    count+=1
                    print(count)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

                elif key == ord('t'):
                    print("Autofocus trigger (and disable continuous)")
                    ctrl = dai.CameraControl()
                    ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
                    ctrl.setAutoFocusTrigger()
                    controlQueue.send(ctrl)


            # t.join()
            # print("FPS: {:.2f}".format(fps.fps()))
            for key in angle_dict1.keys():
                print("The average angle for",key,"is",angle_dict1.get(key))
            print(location_dict1)
            # cap.release()

            # reinitialize for next video
            keypoints_list = None
            detected_keypoints = None
            personwiseKeypoints = None

            angle_dict2={}
            eyes_list=[]
            lkneeflex_list=[]
            rkneeflex_list=[]
            lhipflex_list=[]
            rhipflex_list=[]

            if args.video2:
                count = 0
                frame_count = 0
                frame_stop = None
                location_dict2 = {}

                while should_run2():
                    read_correctly2,frame2=cap2.read()
                    frame_count += 1

                    # restarting video so that the rest of the original video NN
                    # can finish running
                    if frame_count == cap2.get(cv2.CAP_PROP_FRAME_COUNT):
                        frame_count = 0
                        print("restarting video2")
                        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        read_correctly2,frame2=cap2.read()

                    # stop the replayed video when it has done the
                    # amount of frames it skipped at beginning
                    if frame_stop is not None:
                        if frame_count==frame_stop:
                            break

                    h, w = frame2.shape[:2]  # 256, 456
                    debug_frame2 = frame2.copy()

                    nn_data = dai.NNData()
                    nn_data.setLayer("input", to_planar(debug_frame2, (456, 256)))
                    pose_in.send(nn_data)


                    if debug:
                        pos_dict={}
                        if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
                            if frame_stop is None:
                                frame_stop = frame_count
                            for i in range(18):
                                for j in range(len(detected_keypoints[i])):
                                    cv2.circle(debug_frame2, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                                    dict = {keypointsMapping[i]: detected_keypoints[i][j][0:2]}
                                    pos_dict.update(dict)

                            for i in range(0,len(keypointsMapping)):
                                if keypointsMapping[i] in pos_dict.keys():
                                    current_pos = pos_dict.get(keypointsMapping[i])

                                    if keypointsMapping[i] in location_dict2.keys():
                                        previous_pos = location_dict2.pop(keypointsMapping[i])
                                        previous_posx = previous_pos[0]
                                        previous_posy = previous_pos[1]

                                        if isinstance(previous_posx,list):
                                            previous_posx.append(current_pos[0])
                                            previous_posy.append(current_pos[1])
                                        else:
                                            previous_posx = [previous_posx]
                                            previous_posy = [previous_posy]
                                            previous_posx.append(current_pos[0])
                                            previous_posy.append(current_pos[1])

                                        dict = {keypointsMapping[i]: [previous_posx, previous_posy]}

                                    else:
                                        dict = {keypointsMapping[i]: [current_pos[0], current_pos[1]]}
                                else:
                                    if keypointsMapping[i] in location_dict2.keys():
                                        previous_pos = location_dict2.pop(keypointsMapping[i])
                                        previous_posx = previous_pos[0]
                                        previous_posy = previous_pos[1]

                                        if isinstance(previous_posx,list):
                                            previous_posx.append(-1)
                                            previous_posy.append(-1)
                                        else:
                                            previous_posx = [previous_posx]
                                            previous_posy = [previous_posy]
                                            previous_posx.append(-1)
                                            previous_posy.append(-1)

                                        dict = {keypointsMapping[i]: [previous_posx, previous_posy]}

                                    else:
                                        dict = {keypointsMapping[i]: [-1, -1]}
                                location_dict2.update(dict)

                            if 'Nose' in pos_dict.keys() and 'R-Eye' in pos_dict.keys() and 'L-Eye' in pos_dict.keys():
                                angle = getAngle(pos_dict.get('L-Eye'),pos_dict.get('Nose'),pos_dict.get('R-Eye'))
                                eyes_list.append(angle)
                                dict = {'EyesAvg': np.mean(eyes_list)}
                                angle_dict2.update(dict)
                                dict = {'Eyes': eyes_list}
                                angle_dict2.update(dict)
                                print("Eyes Angle", angle)

                            else:
                                eyes_list.append(-1000)
                                angle_dict2.update({'Eyes':eyes_list})

                            if 'L-Hip' in pos_dict.keys() and 'L-Knee' in pos_dict.keys() and 'L-Ank' in pos_dict.keys():
                                angle = getAngle(pos_dict.get('L-Hip'),pos_dict.get('L-Knee'),pos_dict.get('L-Ank'))
                                lkneeflex_list.append(angle)
                                dict = {'LKneeFlexAvg': np.mean(lkneeflex_list)}
                                angle_dict2.update(dict)
                                dict = {'LKneeFlex': lkneeflex_list}
                                angle_dict2.update(dict)
                                print("Left Knee Flexion", angle)

                            else:
                                lkneeflex_list.append(-1000)
                                angle_dict2.update({'LKneeFlex':lkneeflex_list})

                            if 'R-Hip' in pos_dict.keys() and 'R-Knee' in pos_dict.keys() and 'R-Ank' in pos_dict.keys():
                                angle = getAngle(pos_dict.get('R-Hip'),pos_dict.get('R-Knee'),pos_dict.get('R-Ank'))
                                rkneeflex_list.append(angle)
                                dict = {'RKneeFlexAvg': np.mean(rkneeflex_list)}
                                angle_dict2.update(dict)
                                dict = {'RKneeFlex': rkneeflex_list}
                                angle_dict2.update(dict)
                                print("Right Knee Flexion", angle)

                            else:
                                rkneeflex_list.append(-1000)
                                angle_dict2.update({'RKneeFlex':rkneeflex_list})

                            if  'L-Sho' in pos_dict.keys() and 'L-Hip' in pos_dict.keys() and 'L-Knee' in pos_dict.keys():
                                angle = getAngle(pos_dict.get('L-Sho'), pos_dict.get('L-Hip'),pos_dict.get('L-Knee'))
                                lhipflex_list.append(angle)
                                dict = {'LHipFlexAvg': np.mean(lhipflex_list)}
                                angle_dict2.update(dict)
                                dict = {'LHipFlex': lhipflex_list}
                                angle_dict2.update(dict)
                                print("Left Hip Flexion", angle)
                            else:
                                lhipflex_list.append(-1000)
                                angle_dict2.update({'LHipFlex':lhipflex_list})

                            if  'R-Sho' in pos_dict.keys() and 'R-Hip' in pos_dict.keys() and 'R-Knee' in pos_dict.keys():
                                angle = getAngle(pos_dict.get('R-Sho'), pos_dict.get('R-Hip'),pos_dict.get('R-Knee'))
                                rhipflex_list.append(angle)
                                dict = {'RHipFlexAvg': np.mean(rhipflex_list)}
                                angle_dict2.update(dict)
                                dict = {'RHipFlex': rhipflex_list}
                                angle_dict2.update(dict)
                                print("Right Hip Flexion", angle)
                            else:
                                rhipflex_list.append(-1000)
                                angle_dict2.update({'RHipFlex':rhipflex_list})

                            for i in range(17):
                                for n in range(len(personwiseKeypoints)):
                                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                                    if -1 in index:
                                        continue
                                    B = np.int32(keypoints_list[index.astype(int), 0])
                                    A = np.int32(keypoints_list[index.astype(int), 1])
                                    cv2.line(debug_frame2, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

                            fps2.next_iter()

                            cv2.putText(debug_frame2, f"MONO FPS: {round(fps2.fps(), 1)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                            cv2.putText(debug_frame2, f"NN FPS:  {round(fps2.tick_fps('nn'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                            cv2.imshow("mono2", debug_frame2)

                        count+=1
                        print(count)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

                    elif key == ord('t'):
                        print("Autofocus trigger (and disable continuous)")
                        ctrl = dai.CameraControl()
                        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
                        ctrl.setAutoFocusTrigger()
                        controlQueue.send(ctrl)
            # t.join() j
            # print("FPS: {:.2f}".format(fps.fps()))
            # for key in angle_dict2.keys():
            #     print("The average angle for",key,"is",angle_dict2.get(key))
            # cap2.release()
            disparity_x = []
            disparity_y = []

            for key in location_dict1:
                mono1x,mono1y = location_dict1.get(key)
                mono2x,mono2y = location_dict2.get(key)
                for i in range(0,len(mono2x)):
                    if mono2x[i] != -1 and mono1x[i] != -1:
                        disparity_x.append(mono2x[i]-mono1x[i])
                        disparity_y.append(mono2y[i]-mono1y[i])
                    else:
                        disparity_x.append(None)
                        disparity_y.append(None)

            print(disparity_x, disparity_y)



                #print(len(location_dict1.get(key)[0]), len(location_dict2.get(key)[0]))

        except KeyboardInterrupt:
            pass

        running = False

    t.join()
    print("FPS: {:.2f}".format(fps.fps()))
    for key in angle_dict2.keys():
        print("The average angle for",key,"is",angle_dict2.get(key))

    print(location_dict2)
    print(angle_dict1)
    print(angle_dict2)

    if args.video1:
        name = args.video1
        if os.name == "nt":
            name = name.split('\\')
            name = name[-1].split(".mp4")
            name = name[0].split("_")
        else:
            name = name.split('/')
            name = name[-1].split(".mp4")
            name = name[0].split("_")

        if name[2] == "S" and name[1][-1] == "L":
            LHipFlex = np.subtract(np.array(angle_dict1.get("LHipFlex")),180)
            RHipFlex = np.subtract(np.array(angle_dict1.get("RHipFlex")),180)
            LKneeFlex = np.subtract(np.array(angle_dict1.get("LKneeFlex")),90)
            RKneeFlex = np.subtract(np.array(angle_dict1.get("RKneeFlex")),90)
            title = "Sagittal Plane Kinematics"
            subtitle1 = "Hip Flexion"
            subtitle2 = "Knee Flexion"
            YLabel1 = 'Ext     ($^\circ$)      Flex'
            YLabel2 = 'Ext     ($^\circ$)      Flex'
        elif name[2] == "S" and name[1][-1] == "R":
            LHipFlex = np.subtract(np.array(angle_dict1.get("LHipFlex")),180)*-1
            RHipFlex = np.subtract(np.array(angle_dict1.get("RHipFlex")),180)*-1
            LKneeFlex = np.subtract(np.array(angle_dict1.get("LKneeFlex")),180)
            RKneeFlex = np.subtract(np.array(angle_dict1.get("RKneeFlex")),180)
            title = "Sagittal Plane Kinematics"
            subtitle1 = "Hip Flexion"
            subtitle2 = "Knee Flexion"
            YLabel1 = 'Ext     ($^\circ$)      Flex'
            YLabel2 = 'Ext     ($^\circ$)      Flex'

        # normalize for direction
        elif name[2] == "C" and name[1][-1] == "L":
            LHipFlex = np.subtract(np.array(angle_dict1.get("LHipFlex")),180)
            RHipFlex = np.subtract(np.array(angle_dict1.get("RHipFlex")),180)
            LKneeFlex = np.subtract(np.array(angle_dict1.get("LKneeFlex")),180)
            RKneeFlex = np.subtract(np.array(angle_dict1.get("RKneeFlex")),180)
            title = "Coronal Plane Kinematics"
            subtitle1 = "Hip Abd/adduction"
            subtitle2 = "Knee Varus/Valgus"
            YLabel1 = 'Abd     ($^\circ$)      Add'
            YLabel2 = 'Val     ($^\circ$)      Var'
        elif name[2] == "C" and name[1][-1] == "R":
            LHipFlex = np.subtract(np.array(angle_dict1.get("LHipFlex")),180)
            RHipFlex = np.subtract(np.array(angle_dict1.get("RHipFlex")),180)
            LKneeFlex = np.subtract(np.array(angle_dict1.get("LKneeFlex")),180)
            RKneeFlex = np.subtract(np.array(angle_dict1.get("RKneeFlex")),180)
            title = "Coronal Plane Kinematics"
            subtitle1 = "Hip Abd/adduction"
            subtitle2 = "Knee Varus/Valgus"
            YLabel1 = 'Abd     ($^\circ$)      Add'
            YLabel2 = 'Val     ($^\circ$)      Var'

    NormTrialNamePNG = ''.join([name[1],"_",name[2],"_kinematics.png"])
    NormGraphTitle = ' '.join([title, name[1], name[2]])
    plt.figure(figsize=(14, 12))
    #NormGraphTitle=' '.join([NormGraphTitle,"Left GC:",str(LFSActual),"Right GC:",str(RFSActual)])
    plt.suptitle(NormGraphTitle, fontsize=12, fontweight="bold")

    pplot(range(0,len(LHipFlex)),1,LHipFlex,
    RHipFlex, subtitle1, "Time (frames)", YLabel1,
    dict=None, dictkey=None)
    pplot(range(0,len(LKneeFlex)),2,LKneeFlex,
    RKneeFlex, subtitle2, "Time (frames)", YLabel2,
    dict=None, dictkey=None)

    plt.savefig(NormTrialNamePNG)
    plt.close()
    # if args.video1:
    cap.release()
    cap2.release()




# new pipeline for human pose utilizing right mono camera
# TODO: add stereo depth by using other mono camera
elif args.mcamera:
    color = (255, 255, 255)
    fps = FPSHandler()
    # Pipeline defined, now the device is connected to
    with dai.Device(create_pipeline()) as device:
        print("Starting pipeline...")
        # Start pipeline
        device.startPipeline()

        # Output queues will be used to get the grayscale frames and nn data from the outputs defined above
        q_right = device.getOutputQueue("right", 1, blocking=False)
        q_nn_right = device.getOutputQueue("pose_nn_right", 1, blocking=False)
        if depth_bool:
            depthQueue = device.getOutputQueue(name="depth1", maxSize=1, blocking=False)
            spatialCalcQueue = device.getOutputQueue(name="spatialData1", maxSize=1, blocking=False)
            spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig1")
        else:
            q_disparity = device.getOutputQueue("disparity", 1, blocking=False)

        t_right = threading.Thread(target=pose_thread2, args=(q_nn_right, ))
        t_right.start()

        def should_run():
            return cap.isOpened() if args.video1 else True

        frame = None
        bboxes = []
        confidences = []
        labels = []

        angle_dict={}
        eyes_list=[]
        lkneeflex_list=[]
        rkneeflex_list=[]
        try:
            right_frame=None
            debug_right_frame=None
            depthFrameColor=None
            pos_dict={}
            while should_run():
                fps.next_iter()
                #h, w = frame.shape[:2]

                # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
                in_right = q_right.tryGet()
                in_nn_right = q_nn_right.tryGet()
                if not depth_bool:
                    inDepth = q_disparity.tryGet()  # blocking call, will wait until a new data has arrived
                if depth_bool:
                    inDepth = None
                    frame = None
                    inDepth2 = depthQueue.tryGet()
                    inDepthAvg = spatialCalcQueue.tryGet() # Blocking call, will wait until a new data has arrived
                    if inDepth2 is not None and inDepthAvg is not None:
                        depthFrame = inDepth2.getFrame()
                        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                        depthFrameColor = cv2.equalizeHist(depthFrameColor)
                        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                        spatialData = inDepthAvg.getSpatialLocations()

                        for depthData in spatialData:
                            roi = depthData.config.roi
                            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
                            xmin = int(roi.topLeft().x)
                            ymin = int(roi.topLeft().y)
                            xmax = int(roi.bottomRight().x)
                            ymax = int(roi.bottomRight().y)

                            fontType = cv2.FONT_HERSHEY_TRIPLEX
                            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
                            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
                            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)

                if inDepth is not None:
                    #shape = (3, inDepth.getHeight(), inDepth.getWidth())
                    # data is originally represented as a flat 1D array, it needs to be converted into HxW form
                    # frame = inDepth.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                    #frame = np.array(inDepth.getData()).astype(np.uint8).view(np.uint16).reshape((inDepth.getHeight(), inDepth.getWidth()))

                    frame = inDepth.getData().reshape((inDepth.getHeight(), inDepth.getWidth())).astype(np.uint8)
                    frame = np.ascontiguousarray(frame)

                    # frame is transformed, the color map will be applied to highlight the depth info
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                    # frame is ready to be shown


                    # keep the opencv drawings even if the neural network
                    # has not updated
                    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
                        for i in range(18):
                            for j in range(len(detected_keypoints[i])):
                                x, y = detected_keypoints[i][j][0:2]
                                x = int(x*1280/456)
                                y = int(y*720/256)
                                cv2.circle(frame, (x,y), 5, colors[i], -1, cv2.LINE_AA)
                        for i in range(17):
                            for n in range(len(personwiseKeypoints)):
                                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                                if -1 in index:
                                    continue
                                B = np.int32(keypoints_list[index.astype(int), 0])
                                A = np.int32(keypoints_list[index.astype(int), 1])
                                cv2.line(frame, (int(B[0]*1280/456), int(A[0]*1280/456)), (int(B[1]*720/256), int(A[1]*720/256)), colors[i], 3, cv2.LINE_AA)

                    #cv2.imshow("disparity", frame)

                # if there is a frame from the right stereo camera, keep it
                if in_right is not None:
                    shape = (3, in_right.getHeight(), in_right.getWidth())
                    right_frame = in_right.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                    right_frame = np.ascontiguousarray(right_frame)
                    debug_right_frame = right_frame


                    # keep the opencv drawings even if the neural network
                    # has not updated
                    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
                        for i in range(18):
                            for j in range(len(detected_keypoints[i])):
                                cv2.circle(right_frame, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                        for i in range(17):
                            for n in range(len(personwiseKeypoints)):
                                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                                if -1 in index:
                                    continue
                                B = np.int32(keypoints_list[index.astype(int), 0])
                                A = np.int32(keypoints_list[index.astype(int), 1])
                                cv2.line(right_frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

                # if there is a frame from neural network, use it to draw the frames
                if in_nn_right is not None:
                    print("in neuralnet")
                    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
                        for i in range(18):
                            for j in range(len(detected_keypoints[i])):
                                cv2.circle(debug_right_frame, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                                dict = {keypointsMapping[i]: detected_keypoints[i][j][0:2]}
                                pos_dict.update(dict)
                                print(dict)

                        if 'Nose' in pos_dict.keys() and 'R-Eye' in pos_dict.keys() and 'L-Eye' in pos_dict.keys():
                            angle = getAngle(pos_dict.get('L-Eye'),pos_dict.get('Nose'),pos_dict.get('R-Eye'))
                            eyes_list.append(angle)
                            dict = {'Eyes': np.mean(eyes_list)}
                            angle_dict.update(dict)
                            print("Eyes Angle", angle)

                        if 'L-Hip' in pos_dict.keys() and 'L-Knee' in pos_dict.keys() and 'L-Ank' in pos_dict.keys():
                            xlhip=pos_dict.get('L-Hip')[0]
                            ylhip=pos_dict.get('L-Hip')[1]
                            topLeft.y = float((ylhip+1)/256)
                            bottomRight.y = float((ylhip-1)/256)
                            topLeft.x = float((xlhip+1)/456)
                            bottomRight.x = float((xlhip-1)/456)
                            config.roi = dai.Rect(topLeft, bottomRight)
                            cfg = dai.SpatialLocationCalculatorConfig()
                            cfg.addROI(config)
                            spatialCalcConfigInQueue.send(cfg)

                            angle = getAngle(pos_dict.get('L-Hip'),pos_dict.get('L-Knee'),pos_dict.get('L-Ank'))-180
                            lkneeflex_list.append(angle)
                            dict = {'LKneeFlex': np.mean(lkneeflex_list)}
                            angle_dict.update(dict)
                            print("Left Knee Flexion", angle)

                        if 'R-Hip' in pos_dict.keys() and 'R-Knee' in pos_dict.keys() and 'R-Ank' in pos_dict.keys():
                            xrhip=pos_dict.get('R-Hip')[0]
                            yrhip=pos_dict.get('R-Hip')[1]
                            topLeft.y = float((yrhip+1)/256)
                            bottomRight.y = float((yrhip-1)/256)
                            topLeft.x =float((xrhip+1)/456)
                            bottomRight.x = float((xrhip-1)/456)
                            config.roi = dai.Rect(topLeft, bottomRight)
                            cfg = dai.SpatialLocationCalculatorConfig()
                            cfg.addROI(config)
                            spatialCalcConfigInQueue.send(cfg)

                            angle = getAngle(pos_dict.get('R-Hip'),pos_dict.get('R-Knee'),pos_dict.get('R-Ank'))-180
                            rkneeflex_list.append(angle)
                            dict = {'RKneeFlex': np.mean(rkneeflex_list)}
                            angle_dict.update(dict)
                            print("Right Knee Flexion", angle)


                        for i in range(17):
                            for n in range(len(personwiseKeypoints)):
                                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                                if -1 in index:
                                    continue
                                B = np.int32(keypoints_list[index.astype(int), 0])
                                A = np.int32(keypoints_list[index.astype(int), 1])
                                cv2.line(debug_right_frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

                # if there are frames, draw them in real time to the users
                if right_frame is not None:
                    cv2.putText(right_frame, f"MONO FPS: {round(fps.fps(), 1)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.putText(right_frame, f"NN FPS:  {round(fps.tick_fps('nn'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.imshow("right mono", right_frame)

                if frame is not None:
                    cv2.putText(frame, f"MONO FPS: {round(fps.fps(), 1)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.putText(frame, f"NN FPS:  {round(fps.tick_fps('nn'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.imshow("disparity", frame)

                if depthFrameColor is not None:
                    cv2.imshow("depth", depthFrameColor)


                if cv2.waitKey(1) == ord('q'):
                    break
                elif cv2.waitKey(1) == ord('t'):
                    print("Autofocus trigger (and disable continuous)")
                    ctrl = dai.CameraControl()
                    ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
                    ctrl.setAutoFocusTrigger()
                    controlQueue.send(ctrl)
        except KeyboardInterrupt:
            for key in angle_dict.keys():
                print("The average angle for",key,"is",angle_dict.get(key))
            pass
        running = False
    t.join()
    print("FPS: {:.2f}".format(fps.fps()))
