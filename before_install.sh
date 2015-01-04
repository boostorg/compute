#! /bin/bash
set -ev
if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
    sudo apt-get update -qq
    sudo apt-get install -qq fglrx=2:8.960-0ubuntu1 opencl-headers libboost-chrono1.48-dev libboost-date-time1.48-dev libboost-test1.48-dev libboost-system1.48-dev libboost-filesystem1.48-dev libboost-timer1.48-dev libboost-program-options1.48-dev libboost-thread1.48-dev python-yaml lcov libopencv-dev
    gem install coveralls-lcov
elif [ "${TRAVIS_OS_NAME}" = "osx" ]; then
    brew update
    brew outdated boost || brew upgrade boost
    brew install lcov homebrew/science/opencv
    gem install coveralls-lcov
fi
