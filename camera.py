#!/usr/bin/env python
# coding: utf-8
'''
Created on 2017-10-25

@author: 
'''

from ImageConvert import *
from MVSDK import *
import struct
import time
import datetime
import numpy
import cv2
import gc
import numpy as np

g_cameraStatusUserInfo = b"statusInfo"

# camera connection status change callback
def deviceLinkNotify(connectArg, linkInfo):
    if ( EVType.offLine == connectArg.contents.m_event ):
        print("camera has off line, userInfo [%s]" %(c_char_p(linkInfo).value))
    elif ( EVType.onLine == connectArg.contents.m_event ):
        print("camera has on line, userInfo [%s]" %(c_char_p(linkInfo).value))
    

connectCallBackFuncEx = connectCallBackEx(deviceLinkNotify)

# subscribe camera connection status change
def subscribeCameraStatus(camera):
    # subscribe connection status notify
    eventSubscribe = pointer(GENICAM_EventSubscribe())
    eventSubscribeInfo = GENICAM_EventSubscribeInfo()
    eventSubscribeInfo.pCamera = pointer(camera)
    nRet = GENICAM_createEventSubscribe(byref(eventSubscribeInfo), byref(eventSubscribe))
    if ( nRet != 0):
        print("create eventSubscribe fail!")
        return -1
    
    nRet = eventSubscribe.contents.subscribeConnectArgsEx(eventSubscribe, connectCallBackFuncEx, g_cameraStatusUserInfo)
    if ( nRet != 0 ):
        print("subscribeConnectArgsEx fail!")
        # release subscribe resource before return
        eventSubscribe.contents.release(eventSubscribe)
        return -1  
    
    # release subscribe resource at the end of use
    eventSubscribe.contents.release(eventSubscribe) 
    return 0

# unsubscribe camera connection status change
def unsubscribeCameraStatus(camera):
    # unsubscribe connection status notify
    eventSubscribe = pointer(GENICAM_EventSubscribe())
    eventSubscribeInfo = GENICAM_EventSubscribeInfo()
    eventSubscribeInfo.pCamera = pointer(camera)
    nRet = GENICAM_createEventSubscribe(byref(eventSubscribeInfo), byref(eventSubscribe))
    if ( nRet != 0):
        print("create eventSubscribe fail!")
        return -1
        
    nRet = eventSubscribe.contents.unsubscribeConnectArgsEx(eventSubscribe, connectCallBackFuncEx, g_cameraStatusUserInfo)
    if ( nRet != 0 ):
        print("unsubscribeConnectArgsEx fail!")
        # release subscribe resource before return
        eventSubscribe.contents.release(eventSubscribe)
        return -1
    
    # release subscribe resource at the end of use
    eventSubscribe.contents.release(eventSubscribe)
    return 0   

# set software trigger
def setSoftTriggerConf(camera):
    # create AcquisitionControl node
    acqCtrlInfo = GENICAM_AcquisitionControlInfo()
    acqCtrlInfo.pCamera = pointer(camera)
    acqCtrl = pointer(GENICAM_AcquisitionControl())
    nRet = GENICAM_createAcquisitionControl(pointer(acqCtrlInfo), byref(acqCtrl))
    if ( nRet != 0 ):
        print("create AcquisitionControl fail!")
        return -1
    
    # set trigger source to Software
    trigSourceEnumNode = acqCtrl.contents.triggerSource(acqCtrl)
    nRet = trigSourceEnumNode.setValueBySymbol(byref(trigSourceEnumNode), b"Software")
    if ( nRet != 0 ):
        print("set TriggerSource value [Software] fail!")
        # release node resource before return
        trigSourceEnumNode.release(byref(trigSourceEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return -1
    
    # release node resource at the end of use
    trigSourceEnumNode.release(byref(trigSourceEnumNode))
    
    # set trigger selector to FrameStart
    trigSelectorEnumNode = acqCtrl.contents.triggerSelector(acqCtrl)
    nRet = trigSelectorEnumNode.setValueBySymbol(byref(trigSelectorEnumNode), b"FrameStart")
    if ( nRet != 0 ):
        print("set TriggerSelector value [FrameStart] fail!")
        # release node resource before return
        trigSelectorEnumNode.release(byref(trigSelectorEnumNode))
        acqCtrl.contents.release(acqCtrl) 
        return -1
     
    # release node resource at the end of use
    trigSelectorEnumNode.release(byref(trigSelectorEnumNode))
    
    # set trigger mode to On
    trigModeEnumNode = acqCtrl.contents.triggerMode(acqCtrl)
    nRet = trigModeEnumNode.setValueBySymbol(byref(trigModeEnumNode), b"On")
    if ( nRet != 0 ):
        print("set TriggerMode value [On] fail!")
        # release node resource before return
        trigModeEnumNode.release(byref(trigModeEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return -1
    
    # release node resource at the end of use
    trigModeEnumNode.release(byref(trigModeEnumNode))
    acqCtrl.contents.release(acqCtrl)
    
    return 0     

# set external trigger
def setLineTriggerConf(camera):
    # create AcquisitionControl node
    acqCtrlInfo = GENICAM_AcquisitionControlInfo()
    acqCtrlInfo.pCamera = pointer(camera)
    acqCtrl = pointer(GENICAM_AcquisitionControl())
    nRet = GENICAM_createAcquisitionControl(pointer(acqCtrlInfo), byref(acqCtrl))
    if ( nRet != 0 ):
        print("create AcquisitionControl fail!")
        return -1
    
    # set trigger source to Line1
    trigSourceEnumNode = acqCtrl.contents.triggerSource(acqCtrl)
    nRet = trigSourceEnumNode.setValueBySymbol(byref(trigSourceEnumNode), b"Line1")
    if ( nRet != 0 ):
        print("set TriggerSource value [Line1] fail!")
        # release node resource before return
        trigSourceEnumNode.release(byref(trigSourceEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return -1
    
    # release node resource at the end of use
    trigSourceEnumNode.release(byref(trigSourceEnumNode))
    
    # set trigger selector to FrameStart
    trigSelectorEnumNode = acqCtrl.contents.triggerSelector(acqCtrl)
    nRet = trigSelectorEnumNode.setValueBySymbol(byref(trigSelectorEnumNode), b"FrameStart")
    if ( nRet != 0 ):
        print("set TriggerSelector value [FrameStart] fail!")
        # release node resource before return
        trigSelectorEnumNode.release(byref(trigSelectorEnumNode))
        acqCtrl.contents.release(acqCtrl) 
        return -1
     
    # release node resource at the end of use
    trigSelectorEnumNode.release(byref(trigSelectorEnumNode))
    
    # set trigger mode to On
    trigModeEnumNode = acqCtrl.contents.triggerMode(acqCtrl)
    nRet = trigModeEnumNode.setValueBySymbol(byref(trigModeEnumNode), b"On")
    if ( nRet != 0 ):
        print("set TriggerMode value [On] fail!")
        # release node resource before return
        trigModeEnumNode.release(byref(trigModeEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return -1

    # release node resource at the end of use
    trigModeEnumNode.release(byref(trigModeEnumNode))
    
    # set trigger activation to RisingEdge
    trigActivationEnumNode = acqCtrl.contents.triggerActivation(acqCtrl)
    nRet = trigActivationEnumNode.setValueBySymbol(byref(trigActivationEnumNode), b"RisingEdge")
    if ( nRet != 0 ):
        print("set TriggerActivation value [RisingEdge] fail!")
        # release node resource before return
        trigActivationEnumNode.release(byref(trigActivationEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return -1
    
    # release node resource at the end of use
    trigActivationEnumNode.release(byref(trigActivationEnumNode))
    acqCtrl.contents.release(acqCtrl)
    return 0 

# open camera
def openCamera(camera):
    # connect camera
    nRet = camera.connect(camera, c_int(GENICAM_ECameraAccessPermission.accessPermissionControl))
    if ( nRet != 0 ):
        print("camera connect fail!")
        return -1
    else:
        print("camera connect success.")
  
    # subscribe camera connection status change
    nRet = subscribeCameraStatus(camera)
    if ( nRet != 0 ):
        print("subscribeCameraStatus fail!")
        return -1

    return 0

# close camera
def closeCamera(camera):
    # unsubscribe camera connection status change
    nRet = unsubscribeCameraStatus(camera)
    if ( nRet != 0 ):
        print("unsubscribeCameraStatus fail!")
        return -1
  
    # disconnect camera
    nRet = camera.disConnect(byref(camera))
    if ( nRet != 0 ):
        print("disConnect camera fail!")
        return -1
    
    return 0    

# set camera ExposureTime
def setExposureTime(camera, dVal):
    # create corresponding property node according to the value type of property, here is doubleNode
    exposureTimeNode = pointer(GENICAM_DoubleNode())
    exposureTimeNodeInfo = GENICAM_DoubleNodeInfo() 
    exposureTimeNodeInfo.pCamera = pointer(camera)
    exposureTimeNodeInfo.attrName = b"ExposureTime"
    nRet = GENICAM_createDoubleNode(byref(exposureTimeNodeInfo), byref(exposureTimeNode))
    if ( nRet != 0 ):
        print("create ExposureTime Node fail!")
        return -1
      
    # set ExposureTime
    nRet = exposureTimeNode.contents.setValue(exposureTimeNode, c_double(dVal))  
    if ( nRet != 0 ):
        print("set ExposureTime value [%f]us fail!"  % (dVal))
        # release node resource before return
        exposureTimeNode.contents.release(exposureTimeNode)
        return -1
    else:
        print("set ExposureTime value [%f]us success." % (dVal))
            
    # release node resource at the end of use
    exposureTimeNode.contents.release(exposureTimeNode)    
    return 0
    
# enumerate camera
def enumCameras():
    # get system instance
    system = pointer(GENICAM_System())
    nRet = GENICAM_getSystemInstance(byref(system))
    if ( nRet != 0 ):
        print("getSystemInstance fail!")
        return None, None

    # discover camera 
    cameraList = pointer(GENICAM_Camera()) 
    cameraCnt = c_uint()
    nRet = system.contents.discovery(system, byref(cameraList), byref(cameraCnt), c_int(GENICAM_EProtocolType.typeAll));
    if ( nRet != 0 ):
        print("discovery fail!")
        return None, None
    elif cameraCnt.value < 1:
        print("discovery no camera!")
        return None, None
    else:
        print("cameraCnt: " + str(cameraCnt.value))
        return cameraCnt.value, cameraList

def grabOne(camera):
    # create stream source object
    streamSourceInfo = GENICAM_StreamSourceInfo()
    streamSourceInfo.channelId = 0
    streamSourceInfo.pCamera = pointer(camera)
      
    streamSource = pointer(GENICAM_StreamSource())
    nRet = GENICAM_createStreamSource(pointer(streamSourceInfo), byref(streamSource))
    if ( nRet != 0 ):
        print("create StreamSource fail!")     
        return -1
    
    # create AcquisitionControl node
    acqCtrlInfo = GENICAM_AcquisitionControlInfo()
    acqCtrlInfo.pCamera = pointer(camera)
    acqCtrl = pointer(GENICAM_AcquisitionControl())
    nRet = GENICAM_createAcquisitionControl(pointer(acqCtrlInfo), byref(acqCtrl))
    if ( nRet != 0 ):
        print("create AcquisitionControl fail!")
        # release stream source object before return
        streamSource.contents.release(streamSource)  
        return -1
        
    # execute software trigger once
    trigSoftwareCmdNode = acqCtrl.contents.triggerSoftware(acqCtrl)
    nRet = trigSoftwareCmdNode.execute(byref(trigSoftwareCmdNode))
    if( nRet != 0 ):
        print("Execute triggerSoftware fail!")
        # release node resource before return
        trigSoftwareCmdNode.release(byref(trigSoftwareCmdNode))
        acqCtrl.contents.release(acqCtrl)
        streamSource.contents.release(streamSource)   
        return -1   

    # release node resource at the end of use
    trigSoftwareCmdNode.release(byref(trigSoftwareCmdNode))
    acqCtrl.contents.release(acqCtrl)
    streamSource.contents.release(streamSource) 
    
    return 0  

# set ROI ---Height, width, offsetX, offsetY. Input value shall comply with the step length and Max & Min limits.
def setROI(camera, OffsetX, OffsetY, nWidth, nHeight):
    # get the max width of image
    widthMaxNode = pointer(GENICAM_IntNode())
    widthMaxNodeInfo = GENICAM_IntNodeInfo() 
    widthMaxNodeInfo.pCamera = pointer(camera)
    widthMaxNodeInfo.attrName = b"WidthMax"
    nRet = GENICAM_createIntNode(byref(widthMaxNodeInfo), byref(widthMaxNode))
    if ( nRet != 0 ):
        print("create WidthMax Node fail!")
        return -1
    
    oriWidth = c_longlong()
    nRet = widthMaxNode.contents.getValue(widthMaxNode, byref(oriWidth))
    if ( nRet != 0 ):
        print("widthMaxNode getValue fail!")
        # release node resource before return
        widthMaxNode.contents.release(widthMaxNode)
        return -1  
    
    # release node resource at the end of use
    widthMaxNode.contents.release(widthMaxNode)
    
    # get the max height of image
    heightMaxNode = pointer(GENICAM_IntNode())
    heightMaxNodeInfo = GENICAM_IntNodeInfo() 
    heightMaxNodeInfo.pCamera = pointer(camera)
    heightMaxNodeInfo.attrName = b"HeightMax"
    nRet = GENICAM_createIntNode(byref(heightMaxNodeInfo), byref(heightMaxNode))
    if ( nRet != 0 ):
        print("create HeightMax Node fail!")
        return -1
    
    oriHeight = c_longlong()
    nRet = heightMaxNode.contents.getValue(heightMaxNode, byref(oriHeight))
    if ( nRet != 0 ):
        print("heightMaxNode getValue fail!")
        # release node resource before return
        heightMaxNode.contents.release(heightMaxNode)
        return -1
    
    # release node resource at the end of use
    heightMaxNode.contents.release(heightMaxNode)
        
    # check parameter valid
    if ( ( oriWidth.value < (OffsetX + nWidth)) or ( oriHeight.value < (OffsetY + nHeight)) ):
        print("please check input param!")
        return -1
    
    # set image width
    widthNode = pointer(GENICAM_IntNode())
    widthNodeInfo = GENICAM_IntNodeInfo() 
    widthNodeInfo.pCamera = pointer(camera)
    widthNodeInfo.attrName = b"Width"
    nRet = GENICAM_createIntNode(byref(widthNodeInfo), byref(widthNode))
    if ( nRet != 0 ):
        print("create Width Node fail!") 
        return -1
    
    nRet = widthNode.contents.setValue(widthNode, c_longlong(nWidth))
    if ( nRet != 0 ):
        print("widthNode setValue [%d] fail!" % (nWidth))
        # release node resource before return
        widthNode.contents.release(widthNode)
        return -1  
    
    # release node resource at the end of use
    widthNode.contents.release(widthNode)
    
    # set image height
    heightNode = pointer(GENICAM_IntNode())
    heightNodeInfo = GENICAM_IntNodeInfo() 
    heightNodeInfo.pCamera = pointer(camera)
    heightNodeInfo.attrName = b"Height"
    nRet = GENICAM_createIntNode(byref(heightNodeInfo), byref(heightNode))
    if ( nRet != 0 ):
        print("create Height Node fail!")
        return -1
    
    nRet = heightNode.contents.setValue(heightNode, c_longlong(nHeight))
    if ( nRet != 0 ):
        print("heightNode setValue [%d] fail!" % (nHeight))
        # release node resource before return
        heightNode.contents.release(heightNode)
        return -1    
    
    # release node resource at the end of use
    heightNode.contents.release(heightNode)    
    
    # set OffsetX
    OffsetXNode = pointer(GENICAM_IntNode())
    OffsetXNodeInfo = GENICAM_IntNodeInfo() 
    OffsetXNodeInfo.pCamera = pointer(camera)
    OffsetXNodeInfo.attrName = b"OffsetX"
    nRet = GENICAM_createIntNode(byref(OffsetXNodeInfo), byref(OffsetXNode))
    if ( nRet != 0 ):
        print("create OffsetX Node fail!")
        return -1
    
    nRet = OffsetXNode.contents.setValue(OffsetXNode, c_longlong(OffsetX))
    if ( nRet != 0 ):
        print("OffsetX setValue [%d] fail!" % (OffsetX))
        # release node resource before return
        OffsetXNode.contents.release(OffsetXNode)
        return -1    
    
    # release node resource at the end of use
    OffsetXNode.contents.release(OffsetXNode)  
    
    # set OffsetY
    OffsetYNode = pointer(GENICAM_IntNode())
    OffsetYNodeInfo = GENICAM_IntNodeInfo() 
    OffsetYNodeInfo.pCamera = pointer(camera)
    OffsetYNodeInfo.attrName = b"OffsetY"
    nRet = GENICAM_createIntNode(byref(OffsetYNodeInfo), byref(OffsetYNode))
    if ( nRet != 0 ):
        print("create OffsetY Node fail!")
        return -1
    
    nRet = OffsetYNode.contents.setValue(OffsetYNode, c_longlong(OffsetY))
    if ( nRet != 0 ):
        print("OffsetY setValue [%d] fail!" % (OffsetY))
        # release node resource before return
        OffsetYNode.contents.release(OffsetYNode)
        return -1    
    
    # release node resource at the end of use
    OffsetYNode.contents.release(OffsetYNode)   
    return 0

def getFrame(streamSource):
    # get one frame
    frame = pointer(GENICAM_Frame())
    nRet = streamSource.contents.getFrame(streamSource, byref(frame), c_uint(1000))
    if ( nRet != 0 ):
        print("getFrame fail! Timeout:[1000]ms")
        # release stream source object before return
        streamSource.contents.release(streamSource)   
        return -1 
    else:
        print("getFrame success BlockId = [" + str(frame.contents.getBlockId(frame)) + "], get frame time: " + str(datetime.datetime.now()))
        
    nRet = frame.contents.valid(frame)
    if ( nRet != 0 ):
        print("frame is invalid!")
        # release frame resource before return
        frame.contents.release(frame)
        # release stream source object before return
        streamSource.contents.release(streamSource)
        return -1 

    # fill conversion parameter
    imageParams = IMGCNV_SOpenParam()
    imageParams.dataSize    = frame.contents.getImageSize(frame)
    imageParams.height      = frame.contents.getImageHeight(frame)
    imageParams.width       = frame.contents.getImageWidth(frame)
    imageParams.paddingX    = frame.contents.getImagePaddingX(frame)
    imageParams.paddingY    = frame.contents.getImagePaddingY(frame)
    imageParams.pixelForamt = frame.contents.getImagePixelFormat(frame)

    # copy image data out from frame
    imageBuff = frame.contents.getImage(frame)
    userBuff = c_buffer(b'\0', imageParams.dataSize)
    memmove(userBuff, c_char_p(imageBuff), imageParams.dataSize)

    # release frame resource at the end of use
    frame.contents.release(frame)

    # no format conversion required for Mono8
    if imageParams.pixelForamt == EPixelType.gvspPixelMono8:
        grayByteArray = bytearray(userBuff)
        cvImage = numpy.array(grayByteArray).reshape(imageParams.height, imageParams.width)
    else:
        # convert to BGR24
        rgbSize = c_int()
        rgbBuff = c_buffer(b'\0', imageParams.height * imageParams.width * 3)

        nRet = IMGCNV_ConvertToBGR24(cast(userBuff, c_void_p), \
                                        byref(imageParams), \
                                        cast(rgbBuff, c_void_p), \
                                        byref(rgbSize))

        colorByteArray = bytearray(rgbBuff)
        cvImage = numpy.array(colorByteArray).reshape(imageParams.height, imageParams.width, 3)
    # --- end if ---

    return cvImage

def CameraStatus(camera, status):   
    if status: 
        # open camera
        nRet = openCamera(camera)
        if ( nRet != 0 ):
            print("openCamera fail.")
            return -1
            
        # create stream source object
        streamSourceInfo = GENICAM_StreamSourceInfo()
        streamSourceInfo.channelId = 0
        streamSourceInfo.pCamera = pointer(camera)
        
        streamSource = pointer(GENICAM_StreamSource())
        nRet = GENICAM_createStreamSource(pointer(streamSourceInfo), byref(streamSource))
        if ( nRet != 0 ):
            print("create StreamSource fail!")
            return -1

        # set trigger mode to Off for continuously grabbing
        trigModeEnumNode = pointer(GENICAM_EnumNode())
        trigModeEnumNodeInfo = GENICAM_EnumNodeInfo() 
        trigModeEnumNodeInfo.pCamera = pointer(camera)
        trigModeEnumNodeInfo.attrName = b"TriggerMode"
        nRet = GENICAM_createEnumNode(byref(trigModeEnumNodeInfo), byref(trigModeEnumNode))
        if ( nRet != 0 ):
            print("create TriggerMode Node fail!")
            # release node resource before return
            streamSource.contents.release(streamSource) 
            return -1
        
        nRet = trigModeEnumNode.contents.setValueBySymbol(trigModeEnumNode, b"Off")
        if ( nRet != 0 ):
            print("set TriggerMode value [Off] fail!")
            # release node resource before return
            trigModeEnumNode.contents.release(trigModeEnumNode)
            streamSource.contents.release(streamSource) 
            return -1
        
        # release node resource at the end of use  
        trigModeEnumNode.contents.release(trigModeEnumNode) 

        # start grabbing
        nRet = streamSource.contents.startGrabbing(streamSource, c_ulonglong(0), \
                                                c_int(GENICAM_EGrabStrategy.grabStrartegySequential))
        if( nRet != 0):
            print("startGrabbing fail!")
            # release stream source object before return
            streamSource.contents.release(streamSource)   
            return -1
        return streamSource

    else:
        # stop grabbing
        nRet = streamSource.contents.stopGrabbing(streamSource)
        if ( nRet != 0 ):
            print("stopGrabbing fail!")
            streamSource.contents.release(streamSource)  
            return -1

        # close camera
        nRet = closeCamera(camera)
        if ( nRet != 0 ):
            print("closeCamera fail")
            streamSource.contents.release(streamSource)   
            return -1
        
        # release stream source object at the end of use
        streamSource.contents.release(streamSource)    
        
        return streamSource