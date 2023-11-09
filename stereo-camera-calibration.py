import cv2
import os
import argparse
import numpy as np
import glob
import pickle

def capture_images(left_camera, right_camera):
    save = f'images/{name}'
    if not os._exists(save):
        os.makedirs(save,exist_ok=True)

    capL = cv2.VideoCapture(left_camera)
    capR = cv2.VideoCapture(right_camera)
    cpt = 0
    
    while(True): 
        retL, frameL = capL.read()
        retR, frameR = capR.read()

        if retL and retR:
            stacked = np.hstack((frameL,frameR))
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
                cv2.imwrite(f'images/stereo/left/{cpt}.jpg', frameL)
                cv2.imwrite(f'images/stereo/right/{cpt}.jpg', frameR)
                print('images saved')
                cpt+=1
    
    # After the loop release the cap objects
    capL.release() 
    capR.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows()

def get_corners(fname, rows, columns):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # ret, corners
    return cv2.findChessboardCorners(gray, (rows,columns), None)


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

    height,weidth = 0,0
    for fnameL, fnameR in zip(imagesL, imagesR):
        h,w = grayL.shape
        retL, cornersL = get_corners(fnameL, rows, columns)
        retR, cornersR = get_corners(fnameL, rows, columns)
        
        # If found, add object points, image points (after refining them)
        if retL == True and retR == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (rows,columns), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h,w), None, None)

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

    #capture_images(left_camera, right_camera)
    #calibration(name, rows, columns)