import torch
#torch-->pytorch-->contains dynamic graphs we will be able to compute gradients of composite functions
#in backpropogation
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

#define functions
#frame-->color net--> SSD Neural Network Transform --> format to get into neural network
def detect(frame, net,transform):
    #get the height and width of the image
    height,width=frame.shape[:2]#0->height 1->width
    #frame->torch variable by changing dimensions
    #np array -> torch tensor -->matrix allowed to be fed into neural netowrks
    #fake the dimension -->batch size
    #convert into torch variable 
    frame_t = transform(frame)[0]#frame to numpy array
    x = torch.from_numpy(frame_t).permute(2,0,1)#numpy to torch tensor
    #the above converts RGB(012) to GRB(201) -->Specific to SSD
    #u need to make batches -->since neural network accepts batches of input
    #u do this just before input into neural network
    x=Variable(x.unsqueeze(0))#batch dimension is always th first dimension unsqueeze(0)-->now this has to be converted to torch variable
    y=net(x)
    detections = y.data
    #now to position the detection --> we will take width height of topright and width height bottomleft
    scale=torch.Tensor([width,height,width,height])
    #deetections consists of batch(output),number of classes,number of occurances of the class,tuple[score,x0,y0,x1,y1]
    #score is a treshold outta1 and the cordinates of upperright and bottom left
    for i in range(detections.size(1)):
        j = 0
        #the below for the 0-->batch i-->class j-->number of occurances #detections[bath,class,occurance,score]
        while detections[0,i,j,0] >=0.6:
            pt=(detections[0,i,j,1:]*scale).numpy()
            cv2.rectangle(frame,(int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(255,0,0),2)
            cv2.putText(frame,labelmap[i-1],(int(pt[0]),int(pt[1])),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
            j+=1 
            #to deal with the next occurance of the class
    return frame

#Creating the SSD Neural Network
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',map_location = lambda storage,loc: storage))

#Creating the transformation so reasonable input is given to net
transform=BaseTransform(net.size, (104/256.0,117/256.0,123/256.0) )

#Object Detection!
reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data('fps')
writer = imageio.get_writer('output.mp4',fps = fps)
for i,frame in enumerate(reader):
    frame=detect(frame,net.eval(),transform)
    writer.append_data(frame)
    print(i)
writer.close()



    

            
            
    
    
    
    
    
