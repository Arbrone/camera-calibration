import cv2
import os
import argparse
import numpy as np
from pathlib import Path
import glob
import pickle

def capture_images(device, name):
    save = f'images/{name}'
    if not os._exists(save):
        os.makedirs(save,exist_ok=True)

    # TODO : change by device id or path
    cap = cv2.VideoCapture(0)
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


def calibration(name):
    save = f'{calibration}/name'
    if not os._exists(save):
        os.makedirs(save,exist_ok=True)
        
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('./images/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageSize)

    with open(f'{save}/camera_matrix.pkl', wb) as f:
        pickle.dump(cameraMatrix, cameraMatrix)
    with open(f'{save}/dist_coeffs.pkl', wb) as f:
        pickle.dump(cameraMatrix, distCoeffs)
    with open(f'{save}/rvecs.pkl', wb) as f:
        pickle.dump(cameraMatrix, rvecs)
    with open(f'{save}/tvecs.pkl', wb) as f:
        pickle.dump(cameraMatrix, tvecs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='Camera calibration',
                    description='Tool to calibrate a single acamera given as argument')
    
    parser.add_argument(
        '-d',
        '--device',
        help='The device you want to calibrate (e.g., camera name or identifier)',
        required=True
    )
    parser.add_argument(
        '-n',
        '--name',
        help='A custom name to identify the device',
        required=True
    )

    args = parser.parse_args()
    device = args.device
    name = args.name

    capture_images(device, name)
    calibration()