# Copyright 2019 Google LLC
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
Kubeflow Pipelines MNIST example

Run this script to compile pipeline
"""


import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.onprem as onprem

platform = 'onprem'

@dsl.pipeline(
  name='MNIST',
  description='A pipeline to train and serve the MNIST example.'
)
def mnist_pipeline(model_export_dir='/mnt/export',
                   train_steps='1000',
                   learning_rate='0.01',
                   batch_size='100',
                   output='/mnt'):
  """
  Pipeline with three stages:
    1. train an MNIST classifier
    2. deploy a tf-serving instance to the cluster
    3. deploy a web-ui to interact with it
  """

  if platform != 'GCP':
        vop = dsl.VolumeOp(
            name="create_pvc",
            storage_class="rook-ceph-fs",
            resource_name="pipeline-pvc",
            modes=dsl.VOLUME_MODE_RWM,
            size="1Gi"
        )
        pvc_name = vop.outputs["name"]

        download = dsl.ContainerOp(
            name="download_data",
            image="aiven86/git",
            command=["git", "clone", "https://github.com/cdyangzhenyu/mnist-data.git", str(output) + "/data"],
        ).apply(onprem.mount_pvc(pvc_name, 'local-storage', output))
        download.after(vop)

  train = dsl.ContainerOp(
      name='train',
      image='aiven86/tensorflow-mnist-kfp:1.13.1-gpu',
      arguments=[
          "/opt/model.py",
          "--tf-data-dir", str(output) + "/data",
          "--tf-export-dir", model_export_dir,
          "--tf-train-steps", train_steps,
          "--tf-batch-size", batch_size,
          "--tf-learning-rate", learning_rate
          ]
  ).add_resource_limit("aliyun.com/gpu-mem", 1)
  train.after(download)

  serve_args = [
      '--model-export-path', model_export_dir,
      '--server-name', "mnist-service"
  ]
  if platform != 'GCP':
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
  if platform != 'GCP':
    webui_args.extend([
      '--cluster-name', "mnist-pipeline"
    ])

  web_ui = dsl.ContainerOp(
      name='web-ui',
      image='aiven86/kubeflow-examples_mnist_deploy-service:latest',
      arguments=webui_args
  ).set_image_pull_policy('IfNotPresent')
  web_ui.after(serve)

  steps = [train, serve, web_ui]
  for step in steps:
    if platform == 'GCP':
      step.apply(gcp.use_gcp_secret('user-gcp-sa'))
    else:
      step.apply(onprem.mount_pvc(pvc_name, 'local-storage', output))

if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(mnist_pipeline, __file__ + '.tar.gz')

