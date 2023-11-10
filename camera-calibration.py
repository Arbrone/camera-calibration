import cv2
import os
import argparse
import numpy as np
import glob
import pickle

def gstreamer_pipeline(sensor_id=0, capture_width=1280,capture_height=720,display_width=640,display_height=360,framerate=60,flip_method=0):
    return (
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

def capture_images(device, name):
    save = f'images/{name}'
    if not os._exists(save):
        os.makedirs(save,exist_ok=True)

    cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=device,flip_method=2), cv2.CAP_GSTREAMER)
    cpt = 0
    
    while(True): 
        ret, frame = cap.read() 
        cv2.imshow('frame', frame) 
        key = cv2.waitKey(1)

        # 'q' and 'esc' are used to quit
        if key & 0xFF in [ord('q'), 27]:
            if len(os.listdir(save)) < 10:
                print('At least 10 images are required')
            else:
                break
        # 'enter' is used to saved current frame
        if key & 0xFF == 13: 
            cv2.imwrite(f'{save}/{cpt}.jpg', frame)
            print('image saved')
            cpt+=1
    
    # After the loop release the cap object 
    cap.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 


def calibration(name, rows, columns):
    save = f'calibration/{name}'
    if not os._exists(save):
        os.makedirs(save,exist_ok=True)

    rows = rows
    columns = columns
    world_scaling = 1.
            
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(f'./images/{name}/*')
    height,width = 0,0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (rows,columns), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (rows,columns), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (height,width), None, None)

    with open(f'{save}/camera_matrix.pkl', 'wb') as f:
        pickle.dump(cameraMatrix, f)
    with open(f'{save}/dist_coeffs.pkl', 'wb') as f:
        pickle.dump(distCoeffs, f)
    with open(f'{save}/rvecs.pkl', 'wb') as f:
        pickle.dump(rvecs, f)
    with open(f'{save}/tvecs.pkl', 'wb') as f:
        pickle.dump(tvecs, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='Camera calibration',
                    description='Tool to calibrate a single camera given as argument')
    
    parser.add_argument(
        '-d',
        '--device',
        type=int,
        help='The device you want to calibrate (e.g., camera name or identifier)',
        required=True
    )
    parser.add_argument(
        '-n',
        '--name',
        help='A custom name to identify the device',
        required=True
    )
    parser.add_argument(
        '-r',
        '--rows',
        type=int,
        help='Number of checkerboard rows -1',
        required=True
    )
    parser.add_argument(
        '-c',
        '--columns',
        type=int,
        help='Number of checkerboard columns -1',
        required=True
    )
    parser.add_argument(
        '-s',
        '--size',
        type=float,
        help='Real world square size',
        default=1.
    )

    args = parser.parse_args()
    device = args.device
    name = args.name
    rows = args.rows
    columns = args.columns

    capture_images(device, name)
    calibration(name, rows, columns)

    print(f'Calibration matrices have been saved for {device}')
