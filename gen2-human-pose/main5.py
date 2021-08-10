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
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

################################################################################
# get arguments
################################################################################
parser = argparse.ArgumentParser()

parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-vid1', '--video1', type=str, help="Path to 1st video file to be used for inference")
parser.add_argument('-vid2', '--video2', type=str, help="Path to 2nd video file to be used for inference")


args = parser.parse_args()

if not args.video1:
    raise RuntimeError("No source selected. Please use \"-vid1 <path>\" to run on video")


debug = not args.no_debug

################################################################################
# function definitions
################################################################################
"""
finding minima in flats
https://stackoverflow.com/questions/53466504/finding-singulars-sets-of-local-maxima-minima-in-a-1d-numpy-array-once-again
"""
def local_min_scipy(a):
    minima = argrelextrema(a, np.less_equal, order=10)[0]
    return minima

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
Plotting trajectories
"""
def gplot(time, subplot_pos, left_norm, right_norm, title, xlabel, ylabel,
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
    left_masked = np.ma.masked_where(left_norm < 0, left_norm)
    right_masked = np.ma.masked_where(right_norm < 0, right_norm)
    #plot1 = plt.plot(time, left_norm, '#DC143C', time, right_norm, '#14C108')
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
    # NeuralNetwork
    pose_nn = pipeline.createNeuralNetwork()
    pose_nn.setBlobPath(str(Path("models/human-pose-estimation-0001_openvino_2021.2_8shave.blob").resolve().absolute()))

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
if args.video1:

    if args.video1 and args.video2:
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
        lshoflex_list=[]
        rshoflex_list=[]

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

                        if  'L-Sho' in pos_dict.keys() and 'L-Elb' in pos_dict.keys() and 'L-Hip' in pos_dict.keys():
                            angle = getAngle(pos_dict.get('L-Hip'),pos_dict.get('L-Sho'), pos_dict.get('L-Elb'))
                            lshoflex_list.append(angle)
                            dict = {'LShoFlexAvg': np.mean(lshoflex_list)}
                            angle_dict1.update(dict)
                            dict = {'LShoFlex': lshoflex_list}
                            angle_dict1.update(dict)
                            print("Left Shoulder Flexion", angle)
                        else:
                            lshoflex_list.append(-1000)
                            angle_dict1.update({'LShoFlex':lshoflex_list})

                        if  'R-Sho' in pos_dict.keys() and 'R-Elb' in pos_dict.keys() and 'R-Hip' in pos_dict.keys():
                            angle = getAngle(pos_dict.get('R-Hip'),pos_dict.get('R-Sho'), pos_dict.get('R-Elb'))
                            rshoflex_list.append(angle)
                            dict = {'RShoFlexAvg': np.mean(rshoflex_list)}
                            angle_dict1.update(dict)
                            dict = {'RShoFlex': rshoflex_list}
                            angle_dict1.update(dict)
                            print("Right Shoulder Flexion", angle)
                        else:
                            rshoflex_list.append(-1000)
                            angle_dict1.update({'RShoFlex':rshoflex_list})


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

        except KeyboardInterrupt:
            pass

        running = False

    print(angle_dict1)

    name = args.video1
    if os.name == "nt":
        name = name.split('\\')
        name = name[-1].split(".mp4")
        name = name[0].split("_")
    else:
        name = name.split('/')
        name = name[-1].split(".mp4")
        name = name[0].split("_")





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

        if name[1]=="heelraise":
            Lanktraj = location_dict1.get("L-Ank")
            Ranktraj = location_dict1.get("R-Ank")
            Lanktrajy = savgol_filter(Lanktraj[1], 51, 3)
            Lanktrajx = savgol_filter(Lanktraj[0], 51, 3)
            Ranktrajy = savgol_filter(Ranktraj[1], 51, 3)
            Ranktrajx = savgol_filter(Ranktraj[0], 51, 3)
            num_raiseR = local_min_scipy(Ranktrajy)
            num_raiseL = local_min_scipy(Lanktrajy)
            if len(num_raiseR) <= len(num_raiseL):
                numheelraises = [len(num_raiseR)]
                print(num_raiseR)
            else:
                numheelraises = [len(num_raiseL)]
                print(num_raiseL)
            # Lanktrajy = np.array(Lanktraj[1])
            # Ranktrajy = np.array(Ranktraj[1])
            # Lanktrajx = np.array(Lanktraj[0])
            # Ranktrajx = np.array(Ranktraj[0])

            title = "Trajectories"
            subtitle1 = "Heel Raises Y"
            subtitle2 = "Heel Raises X"
            YLabel1 = "Ankle Y pixel"
            YLabel2 = "Ankle X pixel"

            fig = plt.figure()
            plt.bar(["Heelraises"],numheelraises,color="blue",width = 0.4)
            plt.ylabel('Number')
            plt.title('Repetitions of Task')
            plt.savefig(''.join([name[1],"_",name[2],"_heelraisesbar.png"]))
            plt.close()


            NormTrialNamePNG = ''.join([name[1],"_",name[2],"_trajectories.png"])
            NormGraphTitle = ' '.join([title, name[1], name[2]])
        if name[1] == "ohreachL" or name[1] == "ohreachR" and name[2] == "S":
            LShoFlex = np.subtract(np.array(angle_dict1.get("LShoFlex")),180)
            RShoFlex = np.subtract(np.array(angle_dict1.get("RShoFlex")),180)
            title = "Sagittal Plane Kinematics"
            subtitle1 = "Shoulder Flexion"
            YLabel1 = 'Ext     ($^\circ$)      Flex'
            NormTrialNamePNG = ''.join([name[1],"_",name[2],"_kinematics.png"])
            NormGraphTitle = ' '.join([title, name[1], name[2]])

        elif name[2] == "S" and name[1][-1] == "L":
            LHipFlex = np.subtract(np.array(angle_dict1.get("LHipFlex")),180)
            RHipFlex = np.subtract(np.array(angle_dict1.get("RHipFlex")),180)
            LKneeFlex = np.subtract(np.array(angle_dict1.get("LKneeFlex")),180)*-1
            RKneeFlex = np.subtract(np.array(angle_dict1.get("RKneeFlex")),180)*-1
            title = "Sagittal Plane Kinematics"
            subtitle1 = "Hip Flexion"
            subtitle2 = "Knee Flexion"
            YLabel1 = 'Ext     ($^\circ$)      Flex'
            YLabel2 = 'Ext     ($^\circ$)      Flex'
            NormTrialNamePNG = ''.join([name[1],"_",name[2],"_kinematics.png"])
            NormGraphTitle = ' '.join([title, name[1], name[2]])

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
            NormTrialNamePNG = ''.join([name[1],"_",name[2],"_kinematics.png"])
            NormGraphTitle = ' '.join([title, name[1], name[2]])

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
            NormTrialNamePNG = ''.join([name[1],"_",name[2],"_kinematics.png"])
            NormGraphTitle = ' '.join([title, name[1], name[2]])
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

        elif name[1] == "S2S" and name[2] == "S":
            print("Sit to stand protocol")
            LHipFlex = np.subtract(np.array(angle_dict1.get("LHipFlex")),180)
            RHipFlex = np.subtract(np.array(angle_dict1.get("RHipFlex")),180)
            LKneeFlex = np.subtract(np.array(angle_dict1.get("LKneeFlex")),180)*-1
            RKneeFlex = np.subtract(np.array(angle_dict1.get("RKneeFlex")),180)*-1
            title = "Sagittal Plane Kinematics For Sit to Stands"
            subtitle1 = "Hip Flexion"
            subtitle2 = "Knee Flexion"
            YLabel1 = 'Ext     ($^\circ$)      Flex'
            YLabel2 = 'Ext     ($^\circ$)      Flex'
            NormTrialNamePNG = ''.join([name[1],"_",name[2],"_kinematics.png"])
            NormGraphTitle = ' '.join([title, name[1], name[2]])


    #NormGraphTitle=' '.join([NormGraphTitle,"Left GC:",str(LFSActual),"Right GC:",str(RFSActual)])
    if "ohreachL" or "ohreachR" in name[1]:
        plt.figure(figsize=(14, 12))
        plt.suptitle(NormGraphTitle, fontsize=12, fontweight="bold")

        pplot(range(0,len(LShoFlex)),1,LShoFlex,
        RShoFlex, subtitle1, "Time (frames)", YLabel1,
        dict=None, dictkey=None)

        plt.savefig(NormTrialNamePNG)
        plt.close()

    elif "kinematics" in NormTrialNamePNG:
        plt.figure(figsize=(14, 12))
        plt.suptitle(NormGraphTitle, fontsize=12, fontweight="bold")

        pplot(range(0,len(LHipFlex)),1,LHipFlex,
        RHipFlex, subtitle1, "Time (frames)", YLabel1,
        dict=None, dictkey=None)
        pplot(range(0,len(LKneeFlex)),2,LKneeFlex,
        RKneeFlex, subtitle2, "Time (frames)", YLabel2,
        dict=None, dictkey=None)

        plt.savefig(NormTrialNamePNG)
        plt.close()
    else:
        plt.figure(figsize=(14, 12))
        plt.suptitle(NormGraphTitle, fontsize=12, fontweight="bold")

        gplot(range(0,len(Lanktrajy)),1,Lanktrajy,
        Ranktrajy, subtitle1, "Time (frames)", YLabel1,
        dict=None, dictkey=None)

        gplot(range(0,len(Lanktrajx)),2,Lanktrajx,
        Ranktrajx, subtitle2, "Time (frames)", YLabel2,
        dict=None, dictkey=None)

        plt.savefig(NormTrialNamePNG)
        plt.close()

    Lhiploc = location_dict1.get("L-Hip")
    Rhiploc = location_dict1.get("R-Hip")

    hipcenterlocx = []
    videofps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(videofps))

    for i in range(0,len(Lhiploc[0])):
        if Lhiploc[0][i] != -1 and Rhiploc[0][i] != -1:
            hipcenterlocx.append(np.mean([Lhiploc[0][i],Rhiploc[0][i]]))
        else:
            hipcenterlocx.append(-1)

    hipcenterlocy = []
    for i in range(0,len(Lhiploc[1])):
        if Lhiploc[1][i] != -1 and Rhiploc[1][i] != -1:
            hipcenterlocy.append(np.mean([Lhiploc[1][i],Rhiploc[1][i]]))
        else:
            hipcenterlocy.append(-1)

    indices = [i for i, (x, y) in enumerate(zip(hipcenterlocx,hipcenterlocx[1:])) if x == -1 and y!= -1]
    if hipcenterlocx[0] != -1 and hipcenterlocx[-1] != -1:
        #print((hipcenterlocx[-1]-hipcenterlocx[0]),len(hipcenterlocx))
        velx = (hipcenterlocx[-1]-hipcenterlocx[0])/len(hipcenterlocx)*videofps
    elif hipcenterlocx[0] == -1:
        #print((hipcenterlocx[-1]-hipcenterlocx[indices[0]+1]),(len(hipcenterlocx)-indices[0]+1))
        velx = (hipcenterlocx[-1]-hipcenterlocx[indices[0]+1])/(len(hipcenterlocx)-indices[0]+1)*videofps
    elif hipcenterlocx[-1] == -1:
        #print((hipcenterlocx[indices[-1]-1]-hipcenterlocx[0]),(indices[-1]-1))
        velx = (hipcenterlocx[indices[-1]-1]-hipcenterlocx[0])/(indices[-1]-1)*videofps
    print("average horizontal velocity in pixels/s", velx)
    # if args.video1:
    cap.release()
    #cap2.release()
    ###########################################################################
    # write hip centroid
    ###########################################################################
    CSVName = ''.join([name[1],"_",name[2],"_pelviscentroid.csv"])
    with open(CSVName, 'w') as csvfile:
        fieldnames = ['X','Y']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames,delimiter=',')
        writer.writeheader()
        for i in range(0,len(hipcenterlocx)):
            writer.writerow({
            'X': hipcenterlocx[i],
            'Y':hipcenterlocy[i]
            })
