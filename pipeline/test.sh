for i in `kubectl -n kubeflow get pods | grep mnist| awk '{print $1}'`; do kubectl -n kubeflow describe pod $i | grep Image: | sort |uniq;done | sort | uniq | awk '{print $2}'
