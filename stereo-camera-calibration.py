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

def capture_images(left_camera, right_camera):
    save = f'images/stereo'
    if not os._exists(save):
        os.makedirs(save,exist_ok=True)

    capL = cv2.VideoCapture(gstreamer_pipeline(sensor_id=left_camera,flip_method=2), cv2.CAP_GSTREAMER)
    capR = cv2.VideoCapture(gstreamer_pipeline(sensor_id=right_camera,flip_method=2), cv2.CAP_GSTREAMER)
    cpt = 0
    
    while(True): 
        retL, frameL = capL.read()
        retR, frameR = capR.read()

        if retL and retR:
            height, width = frameL.shape
            stacked = np.hstack((frameL,frameR))
            cv2.imshow('frame', cv2.resize(stacked,(width//2,height//2))) 
            key = cv2.waitKey(1)

            # 'q' and 'esc' are used to quit
            if key & 0xFF in [ord('q'), 27]:
                if cpt < 10:
                    print('At least 10 images are required')
                else:
                    break
            # 'enter' is used to saved current frame
            if key & 0xFF == 13: 
                cv2.imwrite(f'images/stereo/left/{cpt}.jpg', frameL)
                cv2.imwrite(f'images/stereo/right/{cpt}.jpg', frameR)
                print('images saved')
                cpt+=1
    
    # After the loop release the cap objects
    capL.release() 
    capR.release() 
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
    imgpointsL = [] # 2d points in image plane.
    imgpointsR = [] # 2d points in image plane.
    
    imagesL = sorted(glob.glob(f'./images/stereo/left/*'))
    imagesR = sorted(glob.glob(f'./images/stereo/right/*'))

    height,width = 0,0
    for fnameL, fnameR in zip(imagesL, imagesR):
        imgL = cv2.imread(fnameL)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        retL, cornersL = cv2.findChessboardCorners(grayL, (rows,columns), None)

        height, width = grayL.shape

        imgR = cv2.imread(fnameR)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        retR, cornersR = cv2.findChessboardCorners(grayR, (rows,columns), None)
        
        # If found, add object points, image points (after refining them)
        if retL == True and retR == True:
            objpoints.append(objp)
            cornersL = cv2.cornerSubPix(grayL,cornersL, (11,11), (-1,-1), criteria)
            imgpointsL.append(cornersL)

            cornersR = cv2.cornerSubPix(grayL,cornersR, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)

            # Draw and display the corners
            cv2.drawChessboardCorners(imgL, (rows,columns), cornersL, retL)
            cv2.drawChessboardCorners(imgR, (rows,columns), cornersR, retR)
            img = np.hstack((cv2.resize(imgL,(width//2,height//2)), cv2.resize(imgR,(width//2,height//2))))
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    with open('./calibration/D2/camera_matrix.pkl', 'rb') as f:
        mtxL = pickle.load(f)
    with open('./calibration/D2/dist_coeffs.pkl', 'rb') as f:
        distL = pickle.load(f)
    with open('./calibration/J2/camera_matrix.pkl', 'rb') as f:
        mtxR = pickle.load(f)
    with open('./calibration/J2/dist_coeffs.pkl', 'rb') as f:
        distR = pickle.load(f)

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL,
                                                                 mtxR, distR, (width, height), criteria = criteria, flags = stereocalibration_flags)

    print(R)
    print(T)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='Camera calibration',
                    description='Tool to calibrate stereo cameras given as argument')
    
    parser.add_argument(
        '-l',
        '--left',
        required=True
    )
    parser.add_argument(
        '-r',
        '--right',
        required=True
    )
    parser.add_argument(
        '--rows',
        type=int,
        help='Number of checkerboard rows -1',
        required=True
    )
    parser.add_argument(
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
    left_camera = args.left
    right_camera = args.right
    rows = args.rows
    columns = args.columns

    #capture_images(left_camera, right_camera)
    calibration('stereo', rows, columns)