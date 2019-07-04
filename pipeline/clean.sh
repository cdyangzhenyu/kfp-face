kubectl -n kubeflow delete workflow `kubectl -n kubeflow get workflow | grep mnist | awk '{print $1}'`
kubectl -n kubeflow delete deploy mnist-service-v1
kubectl -n kubeflow delete deploy web-ui
kubectl -n kubeflow delete svc web-ui
kubectl -n kubeflow delete svc mnist-service
for i in `kubectl -n kubeflow get pvc | grep mnist | awk '{print $1}'`; do kubectl -n kubeflow delete pvc $i;done
