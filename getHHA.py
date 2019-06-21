# --*-- coding:utf-8 --*--
import cv2, math, os
from tqdm import tqdm
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool
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

def saveHHA(depth, rawDepth, cameraMatrix, outputDir):
    D, RD = getImage(depth, rawDepth)
    camera_matrix = loadCameraParam(cameraMatrix, D.shape)
    # print('max gray value: ', np.max(D))        # make sure that the image is in 'meter'
    hha = getHHA(camera_matrix, D, RD)

    head, tail = os.path.split(depth)
    tail, ext = os.path.splitext(tail)
    cv2.imwrite(os.path.join(outputDir, tail + '.png'), hha)

class Consumer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def processDepthImage(self, depthPath):
        saveHHA(depthPath, None, None, self.output_dir)
        print(depthPath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='HHA from Depth Image')
    parser.add_argument('output_dir', type=str, help='HHA image saved here')
    parser.add_argument('--depth_image', type=str, default=None,
                        help='path to smoothed depth image')
    parser.add_argument('--raw_image', type=str, default=None,
                        help='path to unprocessed depth image')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='directory containing smoothed depth images')
    parser.add_argument('--camera_matrix', type=str, default=None,
                        help='path to camera matrix')

    args = parser.parse_args()

    #MSCOCO
    if args.input_dir is not None:
        processed = [os.path.splitext(f)[0] for f in os.path.listdir(args.output_dir)]
        unprocessed = [os.join(args.input_dir, f) for f in os.path.listdir(args.input_dir) if os.path.splitext(f)[0] not in processed]

        C = Consumer(args.output_dir)
        pool = ThreadPool(4)
        pool.map(C.processDepthImage(), unprocessed)
        pool.close()
        pool.join()

    #SUNRGBD
    elif args.depth_image is not None:
        saveHHA(args.depth_image, args.raw_image, args.camera_matrix, args.output_dir)