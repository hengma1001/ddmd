ps -ef | grep openmm | awk '{print $2}' | xargs kill -9
