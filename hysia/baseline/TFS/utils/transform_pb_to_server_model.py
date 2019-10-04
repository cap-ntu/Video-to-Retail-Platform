import tensorflow as tf
import os

tf.app.flags.DEFINE_integer('training_iteration', 300, 'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '', 'Working directory.')
FLAGS = tf.app.flags.FLAGS


def save_server_models(sess, input, output):
    export_path_base = FLAGS.work_dir
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(input)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(output)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input': tensor_info_x},
            outputs={'output': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'prediction':
                prediction_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save()

    print('Done exporting!')


def transform(model_path, input_tensor_name, output_tensor_name):
    graph = tf.Graph()
    with graph.as_default():
        od_gragh_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_gragh_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_gragh_def, name='')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        input_tensor = graph.get_tensor_by_name(input_tensor_name)
        output_tensor = graph.get_tensor_by_name(output_tensor_name)
        print(input_tensor)
        print(output_tensor)
        save_server_models(sess, input_tensor, output_tensor)
