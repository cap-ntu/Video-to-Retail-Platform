# Author: Wang Yongjie
# Email : yongjie.wang@ntu.edu.sg


import cv2
import sys
sys.path.append("..")
import _init_paths

from models.face_pytorch.recognition.recognition import recog

if __name__ == "__main__":

    model = '../../weights/face_pytorch/backbone_ir50_ms1m_epoch120.pth'
    saved_dataset = 'dataset48-pytorch.pkl'
    pnet = '../../weights/face_pytorch/pnet_epoch.pt'
    onet = '../../weights/face_pytorch/onet_epoch.pt'
    rnet = '../../weights/face_pytorch/rnet_epoch.pt'
    factor = 0.7
    threshold = [0.7, 0.7, 0.9]
    minisize = 25

    test = recog(model, saved_dataset)


    test.init_pytorch_env()
    test.load_feature()
    test.init_mtcnn_detector(pnet, rnet, onet, minisize)

    image = cv2.imread('../test1.jpg')
    rectangles, name_lists, features = test.get_indentity(image, role = True)

    for i in range(len(rectangles)):
        rec = rectangles[i,:]
        cv2.rectangle(image, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])), (0, 0, 255), 1)
        cv2.putText(image, name_lists[i], (int(rec[0]), int(rec[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,color=(152, 255, 204), thickness=2)

    cv2.imwrite('test1-recognition-sphereface-pytorch.jpg', image)

