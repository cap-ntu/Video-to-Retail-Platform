import cv2
import sys
sys.path.append("..")
import _init_paths
from models.face_pytorch.mtcnn.detector import mtcnn_detector

if __name__ == '__main__':

    test = mtcnn_detector(p_model_path="../../weights/face_pytorch/pnet_epoch.pt", r_model_path="../../weights/face_pytorch/rnet_epoch.pt", o_model_path="../../weights/face_pytorch/onet_epoch.pt", minisize = 24, use_cuda=True)
    img = cv2.imread("test1.jpg")
    rectangles, landmarks = test.detect(img)
    # print box_align
    for rec in rectangles:
        cv2.rectangle(img, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])), (0, 0, 255), 1)
    cv2.imwrite('result_mtcnn_pytorch.jpg', img)
