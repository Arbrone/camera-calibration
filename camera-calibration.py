import cv2
import os
import argparse
import numpy as np
import glob
import json
from camera import Camera
from pprint import pprint #TODO : remove import


def capture_images(device, name):
    save = f'images/{name}'
    if not os._exists(save):
        os.makedirs(save,exist_ok=True)

    cap = Camera(device, flip_method=2)
    cpt = 0
    
    while(True): 
        frame = cap.get_frame()
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

    intrinsic_parameters = {
        'camera_matrix': cameraMatrix.tolist(),
        'distortion_coefficients': distCoeffs.tolist()
    }
    #pprint(intrinsic_parameters)

    #TODO : remove [0] when camera will be stable
    extrinsic_parameters = {
        'rotation_vector': rvecs[0].tolist(), #cv2.Rodrigues() to convert rotation vector (1,3) to rotation matrix (3,3)
        'translation_vector': tvecs[0].tolist()
    }
    pprint(extrinsic_parameters)
    
    projection_matrix = np.dot(intrinsic_parameters['camera_matrix'], 
                               np.hstack((extrinsic_parameters['rotation_vector'], extrinsic_parameters['translation_vector']))).tolist()
    pprint(projection_matrix)

    with open(f'./calibration/{name}/calibration_parameters.json', 'w') as file:
        json.dump({'intrinsic': intrinsic_parameters, 
                   'extrinsic': extrinsic_parameters,
                   'projection' : projection_matrix}, file)

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

    #capture_images(device, name)
    calibration(name, rows, columns)

    print(f'Calibration matrices have been saved for {device}')
