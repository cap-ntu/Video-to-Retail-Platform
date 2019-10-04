import _init_paths
import ray
import tensorflow as tf
import cv2
import numpy as np
import os
import os.path as osp
import time
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util


THIS_DIR = osp.dirname(osp.abspath(__file__))

ray.init(temp_dir=THIS_DIR)


@ray.remote(num_gpus=1)
class SSDInception:
    def __init__(self, graph_path, label_path, num_classes):
        import _init_paths
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            # Predefine image size as required by SSD
            self.image_shape = [365, 640, 3]
            # Predefine confidence threshold
            self.thresh = 0.3
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, self.image_shape[0], self.image_shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            self.image_tensor = image_tensor
            self.tensor_dict = tensor_dict

            label_map = label_map_util.load_labelmap(label_path)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                        use_display_name=True)
            self.category_index = label_map_util.create_category_index(categories)

    def detect(self, img):
        if img.shape != self.image_shape:
            img = cv2.resize(img, (self.image_shape[0],self.image_shape[1]))
        # Run inference
        output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: np.expand_dims(img, 0)})
        # All outputs are float32 numpy arrays, so convert types as appropriate
        # Apply threshold on detections
        keep = np.where(output_dict['detection_scores'][0] >= self.thresh)
        output_dict['num_detections'] = keep[0].size
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0][keep].astype(np.uint8).tolist()
        output_dict['detection_classes_names'] = [self.category_index[cls_id]['name'] for cls_id in output_dict['detection_classes']]
        output_dict['detection_boxes'] = (output_dict['detection_boxes'][0][keep]).tolist()
        output_dict['detection_scores'] = (output_dict['detection_scores'][0][keep]).tolist()
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = (output_dict['detection_masks'][0]).tolist()
        return output_dict


if __name__ == "__main__":
    print("pid of the driver is: " + str(os.getpid()))
    # Load detector on remote process
    detector = SSDInception.remote(
        graph_path=osp.join(THIS_DIR, "../weights/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"),
        label_path=osp.join(THIS_DIR, "../third/object_detection/data/mscoco_label_map.pbtxt"),
        num_classes=90,
    )

    print("Detector created")
    # time.sleep(15)
    test_img = cv2.imread("test1.jpg")
    pred_future = detector.detect.remote(test_img)
    res = ray.get(pred_future)
    print(res)
