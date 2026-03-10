#!/bin/bash

if [ $( docker ps -a | grep pysrim | wc -l ) -gt 0 ]; then
    docker stop pysrim > /dev/null 
    docker rm pysrim > /dev/null  
fi

docker run --name pysrim -v "$PWD"/scripts:/opt/pysrim/ -v "$PWD/out":/tmp/output -it costrouc/pysrim sh -c "xvfb-run -a python3.6 /opt/pysrim/F_in_Fe.py $1 $2"
