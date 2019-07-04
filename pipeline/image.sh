docker tag alpine/git aiven86/git
docker tag argoproj/argoexec:v2.3.0 aiven86/argoexec:v2.3.0
docker tag gcr.io/kubeflow-examples/mnist/deploy-service:latest aiven86/kubeflow-examples_mnist_deploy-service
docker tag gcr.io/kubeflow-examples/mnist/model:v20190304-v0.2-176-g15d997b aiven86/kubeflow-examples_mnist_model:v20190304-v0.2-176-g15d997b
docker tag gcr.io/ml-pipeline/ml-pipeline-kubeflow-deployer:7775692adf28d6f79098e76e839986c9ee55dd61 aiven86/ml-pipeline_ml-pipeline-kubeflow-deployer:7775692adf28d6f79098e76e839986c9ee55dd61
docker tag tensorflow/serving:1.11.1 aiven86/tensorflow_serving:1.11.1
docker tag gcr.io/kubeflow-examples/mnist/web-ui:v20190304-v0.2-176-g15d997b-pipelines aiven86/kubeflow-examples_mnist_web-ui:v20190304-v0.2-176-g15d997b-pipelines

docker push aiven86/git
docker push aiven86/argoexec:v2.3.0
docker push aiven86/kubeflow-examples_mnist_deploy-service
docker push aiven86/ml-pipeline_ml-pipeline-kubeflow-deployer:7775692adf28d6f79098e76e839986c9ee55dd61
docker push aiven86/tensorflow_serving:1.11.1
docker push aiven86/kubeflow-examples_mnist_web-ui:v20190304-v0.2-176-g15d997b-pipelines
docker push aiven86/kubeflow-examples_mnist_model:v20190304-v0.2-176-g15d997b
