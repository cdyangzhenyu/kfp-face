import os

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
 

from tensorface.const import PRETREINED_MODEL_DIR

#MODEL_FILE_NAME = '20180402-114759/20180402-114759.pb'
MODEL_FILE_NAME = os.environ.get("MODEL_FILE_NAME")

# to get Flask not complain
global tf
_tf = tf
global sess
sess = None

hostname = os.environ.get("MODEL_FILE_NAME")

tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS
 
def do_inference(hostport):
  """Tests PredictionService with concurrent requests.
  Args:
  hostport: Host:port address of the Prediction Service.
  Returns:
  pred values, ground truth label
  """
  # create connection
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
 
  # initialize a request
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'example_model'
  request.model_spec.signature_name = 'prediction'
 
  # Randomly generate some test data
  temp_data = numpy.random.randn(10, 3).astype(numpy.float32)
  data, label = temp_data, numpy.sum(temp_data * numpy.array([1,2,3]).astype(numpy.float32), 1)
  request.inputs['input'].CopyFrom(
  tf.contrib.util.make_tensor_proto(data, shape=data.shape))
 
  # predict
  result = stub.Predict(request, 5.0) # 5 seconds
  return result, label



def load_model(pb_file, input_map=None):
    global _tf
    global sess
    if sess is None:
        sess = _tf.Session()
        print('Model filename: %s' % pb_file)
        with gfile.FastGFile(pb_file, 'rb') as f:
            graph_def = _tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _tf.import_graph_def(graph_def, input_map=input_map, name='')


load_model(os.path.join(PRETREINED_MODEL_DIR, MODEL_FILE_NAME))


# inception net requires this
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def embedding(face_np):
    global sess
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    x = prewhiten(face_np)
    feed_dict = {images_placeholder: [x], phase_train_placeholder: False}
    result = sess.run(embeddings, feed_dict=feed_dict)[0]
    return result


def input_shape():
    return _tf.get_default_graph().get_tensor_by_name("input:0").get_shape()


def embedding_size():
    return _tf.get_default_graph().get_tensor_by_name("embeddings:0").get_shape()[1]
