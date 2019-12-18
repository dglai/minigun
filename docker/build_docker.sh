#/usr/bin/sh
# From current directory
docker build -t dgllib/minigun-ci-cpu -f Dockerfile.cpu .
docker build -t dgllib/minigun-ci-gpu -f Dockerfile.gpu .