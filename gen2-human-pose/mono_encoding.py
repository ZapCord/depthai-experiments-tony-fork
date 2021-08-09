import cv2
import depthai as dai
import numpy as np
import contextlib

################################################################################
# functions
################################################################################
"""
Synchronizes incoming frames using sequence numbers
"""
def seq(packet):
    return packet.getSequenceNum()

# https://stackoverflow.com/a/10995203/5494277
def has_keys(obj, keys):
    return all(stream in obj for stream in keys)

class PairingSystemDouble:
    allowed_instances = [1, 2, 3, 4]  # Left (1, 3) & Right (2, 4)

    def __init__(self):
        self.seq_packets = {}
        self.last_paired_seq = None

    def add_packet(self, packet):
        if packet is not None and packet.getInstanceNum() in self.allowed_instances:
            seq_key = seq(packet)
            self.seq_packets[seq_key] = {
                **self.seq_packets.get(seq_key, {}),
                packet.getInstanceNum(): packet
            }

    def get_pairs(self):
        results = []
        for key in list(self.seq_packets.keys()):
            if has_keys(self.seq_packets[key], self.allowed_instances):
                results.append(self.seq_packets[key])
                self.last_paired_seq = key
        if len(results) > 0:
            self.collect_garbage()
        return results

    def collect_garbage(self):
        for key in list(self.seq_packets.keys()):
            if key <= self.last_paired_seq:
                del self.seq_packets[key]
class PairingSystem:
    allowed_instances = [1, 2]  # Left (1) & Right (2)

    def __init__(self):
        self.seq_packets = {}
        self.last_paired_seq = None

    def add_packet(self, packet):
        if packet is not None and packet.getInstanceNum() in self.allowed_instances:
            seq_key = seq(packet)
            self.seq_packets[seq_key] = {
                **self.seq_packets.get(seq_key, {}),
                packet.getInstanceNum(): packet
            }

    def get_pairs(self):
        results = []
        for key in list(self.seq_packets.keys()):
            if has_keys(self.seq_packets[key], self.allowed_instances):
                results.append(self.seq_packets[key])
                self.last_paired_seq = key
        if len(results) > 0:
            self.collect_garbage()
        return results

    def collect_garbage(self):
        for key in list(self.seq_packets.keys()):
            if key <= self.last_paired_seq:
                del self.seq_packets[key]
################################################################################
# main
################################################################################
# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color and mono cameras
# colorCam = pipeline.createColorCamera()
cam_left = pipeline.createMonoCamera()
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_right = pipeline.createMonoCamera()
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
# cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)


# Create encoders, one for each camera, consuming the frames and encoding them using H.264 / H.265 encoding
ve1 = pipeline.createVideoEncoder()
#ve1.setDefaultProfilePreset(640, 400, 30, dai.VideoEncoderProperties.Profile.H264_MAIN).
ve1.setDefaultProfilePreset(1280, 720, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
cam_left.out.link(ve1.input)

# ve2 = pipeline.createVideoEncoder()
# ve2.setDefaultProfilePreset(1920, 1080, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)
# colorCam.video.link(ve2.input)

ve3 = pipeline.createVideoEncoder()
#ve3.setDefaultProfilePreset(640, 400, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
ve3.setDefaultProfilePreset(1280, 720, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
cam_right.out.link(ve3.input)

# Create outputs
ve1Out = pipeline.createXLinkOut()
ve1Out.setStreamName('ve1Out')
ve1.bitstream.link(ve1Out.input)

# ve2Out = pipeline.createXLinkOut()
# ve2Out.setStreamName('ve2Out')
# ve2.bitstream.link(ve2Out.input)

ve3Out = pipeline.createXLinkOut()
ve3Out.setStreamName('ve3Out')
ve3.bitstream.link(ve3Out.input)

q_mono_list = []
with contextlib.ExitStack() as stack:
    for device_info in dai.Device.getAllAvailableDevices():
        device = stack.enter_context(dai.Device(pipeline, device_info))
        print("Conected to " + device_info.getMxId())
        device.startPipeline()
        # Output queue will be used to get the rgb frames from the output defined above
        # q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        # q_rgb_list.append(q_rgb)

        # Output queues will be used to get the encoded data from the outputs defined above
        outQ1 = device.getOutputQueue(name='ve1Out', maxSize=4, blocking=False)
        # outQ2 = dev.getOutputQueue(name='ve2Out', maxSize=30, blocking=True)
        outQ3 = device.getOutputQueue(name='ve3Out', maxSize=4, blocking=False)

        q_mono_list.append(outQ1)
        q_mono_list.append(outQ3)

    if len(q_mono_list) > 2:
        with open('C1M1.h264', 'wb') as fileMono1H264, open('C1M2.h264', 'wb') as fileMono2H264, open('C2M1.h264', 'wb') as fileMono3H264, open('C2M2.h264', 'wb') as fileMono4H264:
        #         print("Press Ctrl+C to stop encoding...")
            while True:
                try:
                    # Empty each queue
                    while q_mono_list[0].has():
                        q_mono_list[0].tryGet().getData().tofile(fileMono1H264)
                    while q_mono_list[1].has():
                        q_mono_list[1].tryGet().getData().tofile(fileMono2H264)
                    while q_mono_list[2].has():
                        q_mono_list[2].tryGet().getData().tofile(fileMono3H264)
                    while q_mono_list[3].has():
                        q_mono_list[3].tryGet().getData().tofile(fileMono4H264)
                except KeyboardInterrupt:
                    # Keyboard interrupt (Ctrl + C) detected
                    break
    else:
        with open('C1M1.h264', 'wb') as fileMono1H264, open('C1M2.h264', 'wb') as fileMono2H264:
        #         print("Press Ctrl+C to stop encoding...")
            while True:
                try:
                    # Empty each queue
                    while q_mono_list[0].has():
                        q_mono_list[0].tryGet().getData().tofile(fileMono1H264)
                    while q_mono_list[1].has():
                        q_mono_list[1].tryGet().getData().tofile(fileMono2H264)
                except KeyboardInterrupt:
                    # Keyboard interrupt (Ctrl + C) detected
                    break

    print("To view the encoded data, convert the stream file (.h264/.h265) into a video file (.mp4), using commands below:")
    cmd = "ffmpeg -framerate 30 -i {} -c copy {}"
    print(cmd.format("C1M1.h264", "C1M1.mp4"))
    print(cmd.format("C1M2.h264", "C1M2.mp4"))
    if len(q_mono_list) > 2:
        print(cmd.format("C2M1.h264", "C2M1.mp4"))
        print(cmd.format("C2M2.h264", "C2M2.mp4"))
    #print(cmd.format("color.h265", "color.mp4"))
