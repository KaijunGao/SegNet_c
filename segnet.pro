QT += core
QT -= gui

CONFIG += c++11

TARGET = segnet
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app
DEFINES += USE_OPENCV


QMAKE_CXXFLAGS += -std=c++11

INCLUDEPATH += /home/em-gkj/devdata/chejian_v2/caffe-chejian/include
#INCLUDEPATH += /home/emdata/xuchen/gitlab/chenjian_v2_debug/3rdlib/caffe-chejian/include/
INCLUDEPATH += /usr/local/include/
INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /usr/local/include/opencv2
INCLUDEPATH += /usr/local/cuda/include
INCLUDEPATH += /usr/local/include/node
INCLUDEPATH += /usr/include/hdf5/serial
INCLUDEPATH +=/home/em-gkj/devdata/chejian_v2/caffe-chejian/build/src
#INCLUDEPATH += /home/emdata/xuchen/gitlab/chenjian_v2_debug/3rdlib/caffe-chejian/myqt_build/debug/include/

LIBS += -L/usr/local/lib
LIBS += -L/usr/local/ssl/lib
LIBS += -L/usr/local/cuda/lib
LIBS += -L/home/em-gkj/devdata/chejian_v2/caffe-chejian/build/lib
#LIBS += -L/home/emdata/xuchen/gitlab/chenjian_v2_debug/3rdlib/caffe-chejian/myqt_build/debug/lib
LIBS += -L/usr/local/cuda-8.0/lib64/
LIBS += -L/usr/lib/x86_64-linux-gnu/hdf5/serial
LIBS += -L/usr/lib/
LIBS += -L/usr/local/cuda-8.0/lib64
LIBS += -L/lib/x86_64-linux-gnu/

LIBS += -lopencv_core
LIBS += -lopencv_imgproc
LIBS += -lopencv_highgui
LIBS += -lopencv_ml
LIBS += -lopencv_video
LIBS += -lopencv_features2d
LIBS += -lopencv_calib3d
LIBS += -lopencv_objdetect
LIBS += -lopencv_imgcodecs
LIBS += -lopencv_videoio
LIBS += -lopencv_flann
LIBS += -lboost_serialization
LIBS += -lboost_system
LIBS += -lboost_filesystem
LIBS += -lglog
LIBS += -lcaffe
LIBS += -lhdf5
LIBS += -lhdf5_hl
LIBS += -lboost_thread
LIBS += -lprotobuf
LIBS += -latlas
LIBS += -lcublas_static
LIBS += -lcudart
LIBS += -lculibos
LIBS += -lcurand_static
LIBS += -lssl3
LIBS += -lpthread
LIBS += -ldl
LIBS += -lrt
LIBS += /usr/lib/x86_64-linux-gnu/libgflags.a

SOURCES += main.cpp

