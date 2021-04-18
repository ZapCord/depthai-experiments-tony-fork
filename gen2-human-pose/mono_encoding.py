import depthai as dai

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color and mono cameras
# colorCam = pipeline.createColorCamera()
cam_left = pipeline.createMonoCamera()
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_right = pipeline.createMonoCamera()
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)


# Create encoders, one for each camera, consuming the frames and encoding them using H.264 / H.265 encoding
ve1 = pipeline.createVideoEncoder()
ve1.setDefaultProfilePreset(640, 400, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
cam_left.out.link(ve1.input)

# ve2 = pipeline.createVideoEncoder()
# ve2.setDefaultProfilePreset(1920, 1080, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)
# colorCam.video.link(ve2.input)

ve3 = pipeline.createVideoEncoder()
ve3.setDefaultProfilePreset(640, 400, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
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


# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as dev:
    # Start pipeline
    dev.startPipeline()

    # Output queues will be used to get the encoded data from the outputs defined above
    outQ1 = dev.getOutputQueue(name='ve1Out', maxSize=60, blocking=True)
    # outQ2 = dev.getOutputQueue(name='ve2Out', maxSize=30, blocking=True)
    outQ3 = dev.getOutputQueue(name='ve3Out', maxSize=60, blocking=True)

    # The .h264 / .h265 files are raw stream files (not playable yet)
    with open('mono1.h264', 'wb') as fileMono1H264, open('mono2.h264', 'wb') as fileMono2H264:
        print("Press Ctrl+C to stop encoding...")
        while True:
            try:
                # Empty each queue
                while outQ1.has():
                    outQ1.get().getData().tofile(fileMono1H264)

                # while outQ2.has():
                #     outQ2.get().getData().tofile(fileColorH265)

                while outQ3.has():
                    outQ3.get().getData().tofile(fileMono2H264)
            except KeyboardInterrupt:
                # Keyboard interrupt (Ctrl + C) detected
                break

    print("To view the encoded data, convert the stream file (.h264/.h265) into a video file (.mp4), using commands below:")
    cmd = "ffmpeg -framerate 30 -i {} -c copy {}"
    print(cmd.format("mono1.h264", "mono1.mp4"))
    print(cmd.format("mono2.h264", "mono2.mp4"))
    #print(cmd.format("color.h265", "color.mp4"))
