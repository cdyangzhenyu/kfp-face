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

PYTHONPATH="/facenet/src"

is_aligned='True'


@dsl.pipeline(
  name='FACE_RECOGNITION',
  description='A pipeline to train and serve the FACE RECOGNITION example.'
)
def face_recognition(train_steps='30',
                     learning_rate='-1',
                     batch_size='1000',
                     dataset_dir='/dataset',
                     output_dir='/output'):
  """
  Pipeline with three stages:
    1. prepare the face recognition align dataset CASIA-WebFace
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

  casia_align_data = str(dataset_dir) + "/data/casia_maxpy_mtcnnalign_182_160/" 
  if is_aligned == 'True':
      raw_dataset = dsl.ContainerOp(
          name="raw_dataset",
          image="aiven86/facenet-dataset-casia-maxpy-clean:tail-2000",
          command=["/bin/sh", "-c", "echo 'begin moving data';mv /data/ %s/;echo 'moving is finished';"
                   % str(dataset_dir)],
       ).apply(onprem.mount_pvc(data_pvc_name, 'dataset-storage', dataset_dir))
      raw_dataset.after(output_vop)
      casia_align_data = str(dataset_dir) + "/data/casia_maxpy_tail_2000_mtcnnalign_182_160" 
      align_dataset_lfw = dsl.ContainerOp(
          name="align_dataset_lfw",
          image="aiven86/facenet-tensorflow:1.13.1-gpu-py3",
          command=["/bin/sh", "-c", "python /facenet/src/align/align_dataset_mtcnn.py %s/data/lfw "
                   "%s/data/lfw_mtcnnalign_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.8"
                   % (str(dataset_dir), str(dataset_dir))],
       ).apply(onprem.mount_pvc(data_pvc_name, 'dataset-storage', dataset_dir))
      align_dataset_lfw.container.add_resource_limit("aliyun.com/gpu-mem", 2)
      align_dataset_lfw.container.add_env_variable(V1EnvVar(name='PYTHONPATH', value=PYTHONPATH))
      align_dataset_lfw.after(raw_dataset)
      align_dataset = dsl.ContainerOp(
          name="align_dataset",
          image="aiven86/facenet-tensorflow:1.13.1-gpu-py3",
          command=["/bin/sh", "-c", "python /facenet/src/align/align_dataset_mtcnn.py %s/data/CASIA-maxpy-clean-tail-2000 "
                   "%s --image_size 182 --margin 44 --random_order --gpu_memory_fraction 0.8" % (str(dataset_dir), str(casia_align_data))],
       ).add_resource_limit("aliyun.com/gpu-mem", 2)
      align_dataset.container.add_env_variable(V1EnvVar(name='PYTHONPATH', value=PYTHONPATH))
      align_dataset.after(align_dataset_lfw)
  else:
      align_dataset = dsl.ContainerOp(
          name="align_dataset",
          image="aiven86/facenet-dataset-casia-mtcnnalign:1.0",
          command=["/bin/sh", "-c", "echo 'begin moving data';mv /data/ %s/;echo 'moving is finished';" % str(dataset_dir)],
       )
      align_dataset.after(output_vop)

  train = dsl.ContainerOp(
      name='train',
      image='aiven86/facenet-tensorflow:1.13.1-gpu-py3',
      command=["/bin/sh", "-c", "python /facenet/src/train_softmax.py --logs_base_dir %s/logs/facenet/ --models_base_dir %s/models/facenet/"
               "--data_dir %s --image_size 160 --model_def models.inception_resnet_v1"
               "--lfw_dir %s/data/lfw_mtcnnpy_160/ --optimizer ADAM --learning_rate %s --max_nrof_epochs %s --keep_probability 0.8"
               "--random_crop --random_flip --use_fixed_image_standardization"
               "--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-4"
               "--embedding_size 512 --lfw_distance_metric 1 --lfw_use_flipped_images --lfw_subtract_mean"
               "--validation_set_split_ratio 0.05 --validate_every_n_epochs 5 --prelogits_norm_loss_factor 5e-4"
               "--epoch_size %s" % (str(output_dir), str(output_dir), str(casia_align_data),
                                    str(dataset_dir), learning_rate, train_steps, batch_size)]
  ).add_resource_limit("aliyun.com/gpu-mem", 2)
  train.after(align_dataset)

  serve_args = [
      '--model-export-path', str(output_dir) + "/models/facenet/",
      '--server-name', "mnist-service"
  ]
  if platform == 'onprem':
    serve_args.extend([
        '--cluster-name', "face-recognition-pipeline",
        '--pvc-name', output_pvc_name
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
      '--cluster-name', "face-recognition-pipeline"
    ])

  web_ui = dsl.ContainerOp(
      name='web-ui',
      image='aiven86/kubeflow-examples_mnist_deploy-service:latest',
      arguments=webui_args
  ).set_image_pull_policy('IfNotPresent')
  web_ui.after(serve)

  steps = [align_dataset, train, serve, web_ui]
  for step in steps:
    step.apply(onprem.mount_pvc(data_pvc_name, 'dataset-storage', dataset_dir))
    step.apply(onprem.mount_pvc(output_pvc_name, 'output-storage', output_dir))

if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(face_recognition, __file__ + '.tar.gz')

