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

print('Using depthai module from: ', dai.__file__)
print('Depthai version installed: ', dai.__version__)

################################################################################
# get arguments
################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-ccam', '--ccamera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid and -mcam)")
parser.add_argument('-mcam', '--mcamera', action="store_true", help="Use DepthAI stereo cameras for inference (conflicts with -vid and -ccam)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

if not args.ccamera and not args.video and not args.mcamera:
    raise RuntimeError("No source selected. Please use either \"-cam\" to use RGB camera as a source or \"-vid <path>\" to run on video")

debug = not args.no_debug

################################################################################
# function definitions
################################################################################
"""
gets angle based on 3 points where b is the center point between a and c.
assumes segments are drawn between ab and bc
"""
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang
"""
flattening opencv arrays
"""
def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()


"""
Old pipeline creation for color camera and video file input from
depthai-experiments/gen2-human-pose
"""
def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

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

    # NeuralNetwork
    print("Creating Human Pose Estimation Neural Network...")
    pose_nn = pipeline.createNeuralNetwork()
    if args.ccamera or args.mcamera:
        pose_nn.setBlobPath(str(Path("models/human-pose-estimation-0001_openvino_2021.2_6shave.blob").resolve().absolute()))
    else:
        pose_nn.setBlobPath(str(Path("models/human-pose-estimation-0001_openvino_2021.2_8shave.blob").resolve().absolute()))
    # Increase threads for detection
    pose_nn.setNumInferenceThreads(2)
    # Specify that network takes latest arriving frame in non-blocking manner
    pose_nn.input.setQueueSize(1)
    pose_nn.input.setBlocking(False)
    pose_nn_xout = pipeline.createXLinkOut()
    pose_nn_xout.setStreamName("pose_nn")
    pose_nn.out.link(pose_nn_xout.input)

    if args.ccamera:
        cam.preview.link(pose_nn.input)
    else:
        pose_in = pipeline.createXLinkIn()
        pose_in.setStreamName("pose_in")
        pose_in.out.link(pose_nn.input)

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
        if args.video:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.time()
            if delay > 0:
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


if args.ccamera or args.mcamera:
    fps = FPSHandler()
else:
    cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
    fps = FPSHandler(cap)

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
if args.ccamera or args.video:

    if args.ccamera:
        fps = FPSHandler()
    else:
        cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
        fps = FPSHandler(cap)

    with dai.Device(create_pipeline()) as device:
        print("Starting pipeline...")
        device.startPipeline()
        if args.ccamera:
            cam_out = device.getOutputQueue("cam_out", 1, True)
            controlQueue = device.getInputQueue('control')
        else:
            pose_in = device.getInputQueue("pose_in")
        pose_nn = device.getOutputQueue("pose_nn", 1, False)
        t = threading.Thread(target=pose_thread1, args=(pose_nn, ))
        t.start()

        def should_run():
            return cap.isOpened() if args.video else True


        def get_frame():
            if args.video:
                return cap.read()
            else:
                return True, np.array(cam_out.get().getData()).reshape((3, 256, 456)).transpose(1, 2, 0).astype(np.uint8)

        angle_dict={}
        eyes_list=[]
        lkneeflex_list=[]
        rkneeflex_list=[]

        try:
            while should_run():
                read_correctly, frame = get_frame()

                if not read_correctly:
                    break

                fps.next_iter()
                h, w = frame.shape[:2]  # 256, 456
                debug_frame = frame.copy()

                if args.video:
                    nn_data = dai.NNData()
                    nn_data.setLayer("input", to_planar(frame, (456, 256)))
                    pose_in.send(nn_data)

                if debug:
                    pos_dict={}
                    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
                        for i in range(18):
                            for j in range(len(detected_keypoints[i])):
                                cv2.circle(debug_frame, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                                dict = {keypointsMapping[i]: detected_keypoints[i][j][0:2]}
                                pos_dict.update(dict)

                        if 'Nose' in pos_dict.keys() and 'R-Eye' in pos_dict.keys() and 'L-Eye' in pos_dict.keys():
                            angle = getAngle(pos_dict.get('L-Eye'),pos_dict.get('Nose'),pos_dict.get('R-Eye'))
                            eyes_list.append(angle)
                            dict = {'Eyes': np.mean(eyes_list)}
                            angle_dict.update(dict)
                            print("Eyes Angle", angle)

                        if 'L-Hip' in pos_dict.keys() and 'L-Knee' in pos_dict.keys() and 'L-Ank' in pos_dict.keys():
                            angle = getAngle(pos_dict.get('L-Hip'),pos_dict.get('L-Knee'),pos_dict.get('L-Ank'))
                            lkneeflex_list.append(angle)
                            dict = {'LKneeFlex': np.mean(lkneeflex_list)}
                            angle_dict.update(dict)
                            print("Left Knee Flexion", angle)

                        if 'R-Hip' in pos_dict.keys() and 'R-Knee' in pos_dict.keys() and 'R-Ank' in pos_dict.keys():
                            angle = getAngle(pos_dict.get('R-Hip'),pos_dict.get('R-Knee'),pos_dict.get('R-Ank'))
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
                                cv2.line(debug_frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

                    cv2.putText(debug_frame, f"RGB FPS: {round(fps.fps(), 1)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.putText(debug_frame, f"NN FPS:  {round(fps.tick_fps('nn'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.imshow("rgb", debug_frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

                elif key == ord('t'):
                    print("Autofocus trigger (and disable continuous)")
                    ctrl = dai.CameraControl()
                    ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
                    ctrl.setAutoFocusTrigger()
                    controlQueue.send(ctrl)

        except KeyboardInterrupt:
            pass

        running = False

    t.join()
    print("FPS: {:.2f}".format(fps.fps()))
    for key in angle_dict.keys():
        print("The average angle for",key,"is",angle_dict.get(key))
    if args.video:
        cap.release()

# new pipeline for human pose utilizing right mono camera
# TODO: add stereo depth by using other mono camera
elif args.mcamera:
    print("Creating pipeline...")
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    print("Creating mono cameras...")
    # Define a source - mono (grayscale) camera
    cam_right = pipeline.createMonoCamera()
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    cam_left = pipeline.createMonoCamera()
    cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    #create a node to produce depth map using disparity output
    depth = pipeline.createStereoDepth()
    depth.setConfidenceThreshold(200)
    # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
    # For depth filtering
    median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
    depth.setMedianFilter(median)

    # Better handling for occlusions:
    depth.setLeftRightCheck(False)
    # Closer-in minimum depth, disparity range is doubled:
    depth.setExtendedDisparity(False)
    # Better accuracy for longer distance, fractional disparity 32-levels:
    depth.setSubpixel(False)

    cam_left.out.link(depth.left)
    cam_right.out.link(depth.right)


    # Define a neural network that will make predictions based on the source frames
    print("Creating Human Pose Estimation Neural Network...")

    if args.ccamera:
        pose_nn = pipeline.createNeuralNetwork()
        pose_nn.setBlobPath(str(Path("models/human-pose-estimation-0001_openvino_2021.2_6shave.blob").resolve().absolute()))
    elif args.mcamera:
        pose_nn_right = pipeline.createNeuralNetwork()
        pose_nn_right.setBlobPath(str(Path("models/human-pose-estimation-0001_openvino_2021.2_6shave.blob").resolve().absolute()))
    else:
        pose_nn.setBlobPath(str(Path("models/human-pose-estimation-0001_openvino_2021.2_8shave.blob").resolve().absolute()))

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
    # Create a node to convert the grayscale frame into the nn-acceptable form
    # manip_disparity = pipeline.createImageManip()
    # manip_disparity.initialConfig.setResize(456, 256)

    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    # manip_disparity.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    # cam_right.out.link(manip_disparity.inputImage)
    # manip_disparity.out.link(pose_nn_right.input)
    # manip_right.out.link(xout_manip_disparity.input)
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

    # Pipeline defined, now the device is connected to
    with dai.Device(pipeline) as device:
        print("Starting pipeline...")
        # Start pipeline
        device.startPipeline()

        # Output queues will be used to get the grayscale frames and nn data from the outputs defined above
        q_right = device.getOutputQueue("right", 1, blocking=False)
        q_nn_right = device.getOutputQueue("pose_nn_right", 1, blocking=False)
        q_disparity = device.getOutputQueue("disparity", 1, blocking=False)
        t_right = threading.Thread(target=pose_thread2, args=(q_nn_right, ))
        t_right.start()

        def should_run():
            return cap.isOpened() if args.video else True

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
            pos_dict={}
            while should_run():
                fps.next_iter()
                #h, w = frame.shape[:2]

                # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
                in_right = q_right.tryGet()
                in_nn_right = q_nn_right.tryGet()
                inDepth = q_disparity.tryGet()  # blocking call, will wait until a new data has arrived

                if inDepth is not None:
                    #shape = (3, inDepth.getHeight(), inDepth.getWidth())
                    # data is originally represented as a flat 1D array, it needs to be converted into HxW form
                    # frame = inDepth.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
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
                    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
                        for i in range(18):
                            for j in range(len(detected_keypoints[i])):
                                cv2.circle(debug_right_frame, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                                dict = {keypointsMapping[i]: detected_keypoints[i][j][0:2]}
                                pos_dict.update(dict)

                        if 'Nose' in pos_dict.keys() and 'R-Eye' in pos_dict.keys() and 'L-Eye' in pos_dict.keys():
                            angle = getAngle(pos_dict.get('L-Eye'),pos_dict.get('Nose'),pos_dict.get('R-Eye'))
                            eyes_list.append(angle)
                            dict = {'Eyes': np.mean(eyes_list)}
                            angle_dict.update(dict)
                            print("Eyes Angle", angle)

                        if 'L-Hip' in pos_dict.keys() and 'L-Knee' in pos_dict.keys() and 'L-Ank' in pos_dict.keys():
                            angle = getAngle(pos_dict.get('L-Hip'),pos_dict.get('L-Knee'),pos_dict.get('L-Ank'))-180
                            lkneeflex_list.append(angle)
                            dict = {'LKneeFlex': np.mean(lkneeflex_list)}
                            angle_dict.update(dict)
                            print("Left Knee Flexion", angle)

                        if 'R-Hip' in pos_dict.keys() and 'R-Knee' in pos_dict.keys() and 'R-Ank' in pos_dict.keys():
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
