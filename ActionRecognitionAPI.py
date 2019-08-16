import torch
import os
from torch.utils.data import TensorDataset, DataLoader
from models import *
from utils import *
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
from PIL import Image
from sort import *

train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#########################     YOLO congfigratioin    ########################################
config_path='config/yolov3.cfg'
weights_path='yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.5
nms_thres=0.4
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.to(device)
model.eval()
data_transforms = transforms.Compose([ transforms.ToTensor()])
if device == torch.device("cpu"):
    Tensor = torch.FloatTensor
else:
    Tensor = torch.cuda.FloatTensor
    
    
###########################################################################################

f_rnn = torch.load('final_f_rnn.pt')
b_rnn = torch.load('final_b_rnn.pt')
cnn_model = torch.load('ensembledModel.jfc_6,200_95e.pt',map_location = 'cpu')
if train_on_gpu:
    f_rnn.cuda()
    b_rnn.cuda()
    cnn_model.cuda()

#########################################################################################
def detect_image(img):
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),

         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

###########################################################################################
def chunk(features,seq_length):
    for i in range(0, len(features)-seq_length, 1):
        yield features[i:i+seq_length]
             
###########################################################################################
def batching_data(cnn_out, sequence_length):
    rNpArr = np.flip(cnn_out.cpu().numpy(),0).copy()   
    reversed_cnn_out = torch.from_numpy(rNpArr)
    
    cnn_out = torch.cat((cnn_out[:sequence_length],cnn_out),0)
    reversed_cnn_out = torch.cat((reversed_cnn_out[:sequence_length],reversed_cnn_out),0)
    
    feature_tensors = torch.stack(list(chunk(cnn_out, sequence_length)))
    reversed_feature_tensors = torch.stack(list(chunk(reversed_cnn_out, sequence_length)))
    
    return feature_tensors, reversed_feature_tensors
    
###########################################################################################   
def get_video_frames(video_path):
    
    frames_tensor = torch.FloatTensor(1,3,256,256).to(device)
    cropped_frames_tensor = torch.FloatTensor(1,3,256,256).to(device)

    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mot_tracker = Sort()
    frames = 0
    box_h = 556
    box_w = 556
    y1 = 300
    x1 = 300
    idx=0
    n = 0

    while(True):
        ret, frame = cap.read()
        
        if not ret:
            break
            
        idx+=1
        frame = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        if detections is not None:
            tracked_objects = mot_tracker.update(detections.cpu())
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            starty = y1-20 if y1>20 else 0    
            startx = x1-20 if x1>20 else 0 
            roi = frame[starty:y1 + box_h+50, startx:x1 + box_w+50,:]
        else:
            starty = y1-20 if y1>20 else 0    
            startx = x1-20 if x1>20 else 0
            roi = frame[starty:y1 + box_h+50, startx:x1 + box_w+50,:]
            
        if idx%2==0:
            frame = data_transforms(cv2.resize(frame, (256, 256))).to(device)
            roi = data_transforms(cv2.resize(roi, (256, 256))).to(device)
            frames_tensor = torch.cat((frames_tensor,frame.unsqueeze(0)),0)
            cropped_frames_tensor = torch.cat((cropped_frames_tensor,roi.unsqueeze(0)),0)
            
    frames_tensor = frames_tensor[1:]
    cropped_frames_tensor = cropped_frames_tensor[1:]
    return frames_tensor, cropped_frames_tensor, count

###########################################################################################
def CNN_pass(frames_tensor,cropped_frames_tensor):   
    
    cnn_out200 = torch.FloatTensor(1,200).to(device)
    cnn_out6 = torch.FloatTensor(1,6).to(device)
    cnn_model.eval()
    with torch.no_grad():
        for frame, cropped_frame in zip(frames_tensor,cropped_frames_tensor):
            if train_on_gpu:
                frame  , cropped_frame  = frame.unsqueeze(0).cuda(), cropped_frame.unsqueeze(0).cuda()
            out6,out200 = cnn_model.forward(frame,cropped_frame)
            cnn_out6 = torch.cat((cnn_out6,out6),0)
            cnn_out200 = torch.cat((cnn_out200,out200),0)

    cnn_out6    = cnn_out6[1:]
    cnn_out200  = cnn_out200[1:]
    return cnn_out6, cnn_out200

###########################################################################################
def RNN_pass(data_sequences, reversed_sequences):
    
    hiddenf = f_rnn.init_hidden(data_sequences.size(0))
    hiddenb = b_rnn.init_hidden(data_sequences.size(0))
    
    f_rnn.eval()
    b_rnn.eval()  
    
    with torch.no_grad():
        outputf, hiddenf = f_rnn(data_sequences.to(device)    , hiddenf)
        outputb, hiddenb = b_rnn(reversed_sequences.to(device), hiddenb)
        
    outputf = F.log_softmax(outputf, dim=1)
    outputb = F.log_softmax(outputb, dim=1)
    
    return outputf, outputb

###########################################################################################
def inferOnVideo(video_path):
    
    sequence_length = 75
    frames_tensor,cropped_frames_tensor, count =  get_video_frames(video_path)
    print(count)
    cnn_out6,  cnn_out200 = CNN_pass(frames_tensor,cropped_frames_tensor)
    data_sequences, reversed_sequences = batching_data(cnn_out200,sequence_length)
    outputf, outputb = RNN_pass(data_sequences, reversed_sequences)
    rNpArr = np.flip(outputb.cpu().numpy(),0).copy()   
    outputb = torch.from_numpy(rNpArr).to(device)
    final_out = (outputf + outputb) / 2
    _, pred = torch.max(final_out, 1)
    final_pred = torch.zeros(count)
    
    for i,p in enumerate(pred):
        final_pred[i*2:i*2+2] = p
        
    return final_pred.numpy()
    

























































