# Copyright 2019 Awcloud Co.,Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Kubeflow Pipelines face-recognition example

Run this script to compile pipeline
"""


import kfp.dsl as dsl
import kfp.onprem as onprem
from kubernetes.client.models import V1EnvVar

platform = 'onprem'
is_preprocessed = False

PYTHONPATH="/facenet/src"

@dsl.pipeline(
  name='FACE_RECOGNITION',
  description='A pipeline to train and serve the FACE RECOGNITION example.'
)
def face_recognition(model_export_dir='/output/export',
                     train_steps='1000',
                     learning_rate='0.01',
                     batch_size='100',
                     dataset_dir='/data',
                     output_dir='/output'):
  """
  Pipeline with three stages:
    1. prepare the face recognition dataset CASIA-WebFace
    2. train an facenet classifier model
    3. deploy a tf-serving instance to the cluster
    4. deploy a web-ui to interact with it
  """

  if platform == 'onprem':
        data_vop = dsl.VolumeOp(
            name="prepare_data_vop",
            storage_class="rook-ceph-fs",
            resource_name="face-recognition-data-pvc",
            modes=dsl.VOLUME_MODE_RWM,
            size="10Gi"
        )
        data_pvc_name = data_vop.outputs["name"]
        
        output_vop = dsl.VolumeOp(
            name="prepare_output_vop",
            storage_class="csi-s3",
            resource_name="face-recognition-output-pvc",
            modes=dsl.VOLUME_MODE_RWM,
            size="1Gi"
        )
        output_vop.after(data_vop)      
        output_pvc_name = output_vop.outputs["name"]

  if not is_preprocessed:
      raw_dataset = dsl.ContainerOp(
          name="raw_dataset",
          image="aiven86/facenet-dataset-casia-webface:1.0",
          command=["cp", "-r", "/dataset", str(dataset_dir) + "/dataset"],
       ).apply(onprem.mount_pvc(data_pvc_name, 'local-storage', dataset_dir))
      raw_dataset.after(output_vop)
      align_dataset = dsl.ContainerOp(
          name="align_dataset",
          image="aiven86/facenet-mtcnn-align-process:1.0",
          command=["python", 
                   "/facenet/src/align/align_dataset_mtcnn.py", 
                   str(dataset_dir) + "/dataset/lfw", 
                   str(dataset_dir) + "/dataset/lfw_mtcnnpy_160",
                   "--image_size", "160",
                   "--margin", "32",
                   "--random_order"],
       ) 
      align_dataset.container.add_env_variable(V1EnvVar(name='PYTHONPATH', value=PYTHONPATH))
      align_dataset.after(raw_dataset)
  else:
      align_dataset = dsl.ContainerOp(
          name="align_dataset",
          image="aiven86/facenet-dataset-casia-webface-mtcnn-align:1.0",
          command=["cp", "-r", "/dataset", str(dataset_dir) + "/dataset"],
       )
      align_dataset.after(output_vop)


  train = dsl.ContainerOp(
      name='train',
      image='aiven86/kubeflow-examples_mnist_model:v20190304-v0.2-176-g15d997b',
      arguments=[
          "/opt/model.py",
          "--tf-data-dir", str(output) + "/data",
          "--tf-export-dir", model_export_dir,
          "--tf-train-steps", train_steps,
          "--tf-batch-size", batch_size,
          "--tf-learning-rate", learning_rate
          ]
  )
  train.after(download)

  serve_args = [
      '--model-export-path', model_export_dir,
      '--server-name', "mnist-service"
  ]
  if platform == 'onprem':
    serve_args.extend([
        '--cluster-name', "mnist-pipeline",
        '--pvc-name', pvc_name
    ])

  serve = dsl.ContainerOp(
      name='serve',
      image='aiven86/ml-pipeline_ml-pipeline-kubeflow-deployer:'
            '7775692adf28d6f79098e76e839986c9ee55dd61',
      arguments=serve_args
  )
  serve.after(train)


  webui_args = [
          '--image', 'aiven86/kubeflow-examples_mnist_web-ui:'
                     'v20190304-v0.2-176-g15d997b-pipelines',
          '--name', 'web-ui',
          '--container-port', '5000',
          '--service-port', '80',
          '--service-type', "NodePort"
  ]
  if platform == 'onprem':
    webui_args.extend([
      '--cluster-name', "mnist-pipeline"
    ])

  web_ui = dsl.ContainerOp(
      name='web-ui',
      image='aiven86/kubeflow-examples_mnist_deploy-service:latest',
      arguments=webui_args
  ).set_image_pull_policy('IfNotPresent')
  web_ui.after(serve)

  steps = [align_dataset, train, serve, web_ui]
  for step in steps:
    step.apply(onprem.mount_pvc(data_pvc_name, 'local-storage', dataset_dir))

if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(face_recognition, __file__ + '.tar.gz')

