import depthai as dai
import cv2

# 1. Create the DepthAI pipeline
pipeline = dai.Pipeline()

# 2. Create the color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480) # Set resolution for the preview
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# 3. Create an output node to send frames to your PC
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")

# 4. Link the camera to the output
cam_rgb.preview.link(xout_rgb.input)

# 5. Connect to the device and run the pipeline
print("Searching for OAK-D Lite...")
with dai.Device(pipeline) as device:
    print(f"Successfully connected! Device MxId: {device.getMxId()}")
    print("Press 'q' in the video window to quit.")
    
    # Create a queue to receive the frames
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    
    while True:
        in_rgb = q_rgb.get()          # Grab the packet from the queue
        frame = in_rgb.getCvFrame()   # Convert it to an OpenCV image
        
        # Display the frame
        cv2.imshow("OAK-D Lite Color Feed", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

# Clean up windows when done
cv2.destroyAllWindows()