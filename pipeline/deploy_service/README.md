kubectl run --image=gcr.io/kubeflow-examples/mnist/deploy-service:latest --command -- tail -f /etc/resolv.conf

docker commit xxx gcr.io/kubeflow-examples/mnist/deploy-service:latest

sh build.sh
