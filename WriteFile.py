
# coding: utf-8

# In[9]:


import numpy as np
import os
from SimilarityNormTransform import SimilarityNormTransform
directory = "data"
writingFile = "TrainData"


# In[10]:


# Faulty files
# S001C001P003R001A001 - Num of actors not mentioned in some frames
# S001C002P005R002A002 - Num of actors not mentioned in some frames
# S001C002P005R002A003 - Num of actors not mentioned in some frames


# In[11]:


_spineBase = 0
_neck = 1
_spineMid = 2
_head = 3
_shoulderLeft = 4
_elbowLeft = 5
_wristLeft = 6
_handLeft = 7
_shoulderRight = 8
_elbowRight = 9
_wristRight = 10
_handRight = 11
_hipLeft = 12
_kneeLeft = 13
_ankleLeft = 14
_footLeft = 15
_hipRight = 16
_kneeRight = 17
_ankleRight = 18
_footRight = 19
_spine = 20
_tipHandLeft = 21
_thumbLeft = 22
_tipHandRight = 23
_thumbRight = 24

quads = np.zeros((25, 4), dtype = np.int32)

#Upper Body Quads
quads[0] = _spineBase, _head, _tipHandLeft, _tipHandRight
quads[1] = _spineBase, _spine, _elbowLeft, _elbowRight
quads[2] = _spineBase, _spine, _elbowRight, _tipHandRight
quads[3] = _spineBase, _spine, _elbowLeft, _tipHandLeft
quads[4] = _spineMid, _shoulderLeft, _tipHandLeft, _elbowLeft
quads[5] = _spineMid, _shoulderRight, _tipHandRight, _elbowRight
quads[6] = _shoulderRight, _shoulderLeft, _tipHandLeft, _tipHandRight
quads[7] = _shoulderRight, _elbowRight, _wristRight, _tipHandRight
quads[8] = _shoulderLeft, _elbowLeft, _wristLeft, _tipHandLeft
quads[9] = _shoulderRight, _elbowLeft, _tipHandRight, _tipHandLeft
quads[10] = _shoulderLeft, _elbowRight, _tipHandLeft, _tipHandRight

#Lower Body Quads
quads[11] = _hipRight, _ankleRight, _kneeRight, _kneeLeft
quads[12] = _hipLeft, _ankleLeft, _kneeLeft, _kneeRight
quads[13] = _kneeRight, _ankleRight, _kneeLeft, _ankleLeft
quads[14] = _hipRight, _hipLeft, _ankleRight, _ankleLeft
quads[15] = _kneeRight, _footRight, _kneeLeft, _footLeft

#Upper and Lower Body Mix Quads
quads[16] = _kneeRight, _elbowLeft, _elbowRight, _kneeLeft
quads[17] = _footRight, _shoulderRight, _footLeft, _shoulderLeft
quads[18] = _kneeRight, _kneeLeft, _tipHandRight, _tipHandLeft
quads[19] = _hipRight, _kneeRight, _footRight, _tipHandLeft
quads[20] = _hipLeft, _kneeLeft, _footLeft, _tipHandRight
quads[21] = _shoulderLeft, _elbowLeft, _tipHandLeft, _footRight
quads[22] = _shoulderRight, _elbowRight, _tipHandRight, _footLeft
quads[23] = _elbowLeft, _tipHandLeft, _kneeRight, _footRight
quads[24] = _elbowRight, _tipHandRight, _kneeLeft, _footLeft


# In[12]:


fileList = os.listdir(directory)
print(len(fileList))
for fNum in range (0, len(fileList)):
    #Loop through each file in training set
    print(fNum, fileList[fNum])
    currentLabel = int(fileList[fNum][17] + fileList[fNum][18] + fileList[fNum][19])    
    file = open(directory + "/" + fileList[fNum], 'r')
    firstLine = file.readline()

    firstLine.rstrip('\n')
    numFrames = int(firstLine)

    totalFeatures = np.array((), dtype = np.float32)
    for i in range (0, numFrames):
        frameFeatures = np.array((), dtype = np.float32)

        frameData = np.zeros((25, 3), dtype=np.float32)
        numPersons = int(file.readline().rstrip('\n'))
        file.readline()
        numJoints = int(file.readline().rstrip('\n'))
        if numJoints != 25:
            print("Error! Number of joints is ", numJoints, " in frame ", i)
        for j in range(0, numJoints):
            temp = file.readline()
            jointData = temp.split(' ')
            frameData[j] = jointData[0], jointData[1], jointData[2]
        currentJointSet = np.zeros((4, 3), dtype=np.float32)
        for k in range (0, 25):
            for l in range (0, 4):
                currentJointSet[l] = frameData[quads[k][l]]
            #Joint Set k is now ready
            currentQuad = SimilarityNormTransform(currentJointSet)
            frameFeatures = np.append(frameFeatures, currentQuad)
        #The feature descriptor of this frame is now ready
        totalFeatures = np.append(totalFeatures, frameFeatures)

    #Now total feature descriptor is ready    
    featuresToWrite = ' '.join(str("%0.6f"%e) for e in totalFeatures)
    writeFile = open(writingFile, "a")
    writeFile.write(featuresToWrite + ", " + str(currentLabel-1) + "\n")
    writeFile.close()

    file.close()


# In[5]:


totalFeatures.shape


# In[6]:


str(totalFeatures).translate({ord(c): None for c in '\n[]'})


# In[7]:


tf = np.append(totalFeatures, totalFeatures)
tf.shape


# In[8]:


str1 = ' '.join(str("%0.4f"%e) for e in totalFeatures).translate({ord(c): None for c in '\n[]'})

str1


# In[9]:


totalFeatures

ls


# In[ ]:


import os
lis = os.listdir("data")


# In[ ]:


int(lis[0][17] + lis[0][18] + lis[0][19])


# In[ ]:


f.readline()


# In[ ]:


for m in range (0, 1):
    print(m)

