# --*-- coding:utf-8 --*--
import cv2, os, math
import numpy as np

from utils.rgbd_util import processDepthImage
from utils.getCameraParam import loadCameraParam

'''
must use 'COLOR_BGR2GRAY' here, or you will get a different gray-value with what MATLAB gets.
'''
def getImage(depth_image, raw_depth_image):
    D = cv2.imread(depth_image, cv2.COLOR_BGR2GRAY) / 10000

    if raw_depth_image is None:
        RD = D.copy()
    else:
        RD = cv2.imread(raw_depth_image, cv2.COLOR_BGR2GRAY)/10000
    return D, RD


'''
C: Camera matrix
D: Depth image, the unit of each element in it is "meter"
RD: Raw depth image, the unit of each element in it is "meter"
'''
def getHHA(C, D, RD):
    missingMask = (RD == 0);
    pc, N, yDir, h, pcRot, NRot = processDepthImage(D * 100, missingMask, C);

    tmp = np.multiply(N, yDir)
    acosValue = np.minimum(1,np.maximum(-1,np.sum(tmp, axis=2)))
    angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
    angle = np.reshape(angle, h.shape)

    '''
    Must convert nan to 180 as the MATLAB program actually does. 
    Or we will get a HHA image whose border region is different
    with that of MATLAB program's output.
    '''
    angle[np.isnan(angle)] = 180        


    pc[:,:,2] = np.maximum(pc[:,:,2], 100)
    I = np.zeros(pc.shape)

    # opencv-python save the picture in BGR order.
    I[:,:,2] = 31000/pc[:,:,2]
    I[:,:,1] = h
    I[:,:,0] = (angle + 128-90)

    # print(np.isnan(angle))

    '''
    np.uint8 seems to use 'floor', but in matlab, it seems to use 'round'.
    So I convert it to integer myself.
    '''
    I = np.rint(I)

    # np.uint8: 256->1, but in MATLAB, uint8: 256->255
    I[I>255] = 255
    HHA = I.astype(np.uint8)
    return HHA

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='HHA from Depth Image')
    parser.add_argument('output_dir', type=str, help='HHA image saved here')
    parser.add_argument('depth_image', type=str,
                        help='path to smoothed depth image')
    parser.add_argument('--raw_depth_image', type=str, default=None,
                        help='path to unprocessed depth image')
    parser.add_argument('--camera_matrix', type=str, default=None,
                        help='path to camera matrix')

    args = parser.parse_args()

    D, RD = getImage(args.depth_image, args.raw_depth_image)
    # camera_matrix = getCameraParam()
    camera_matrix = loadCameraParam(args.camera_matrix, D.shape)
    # print('max gray value: ', np.max(D))        # make sure that the image is in 'meter'
    hha = getHHA(camera_matrix, D, RD)

    head, tail = os.path.split(args.depth_image)
    tail, ext = os.path.splitext(tail)
    cv2.imwrite(os.path.join(args.output_dir, tail + '.png'), hha)