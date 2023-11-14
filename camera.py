import cv2

class Camera:
    def __init__(self, sensor_id=0, capture_width=1280,capture_height=720,display_width=640,display_height=360,framerate=60,flip_method=0) -> None:
        self.parameters = (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                capture_width,                        
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )
        self.cap = cv2.VideoCapture(self.parameters, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            raise Exception("Failed to open the device")

    def __str__(self) -> str:
        pass

    def get_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            raise Exception('Failed to capture image')
        
        return frame

    def release(self):
        self.cap.release()