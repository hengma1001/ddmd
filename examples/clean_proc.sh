ps -ef | grep ddmd | awk '{print $2}' | xargs kill -9
