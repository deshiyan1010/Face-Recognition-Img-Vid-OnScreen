from facenet_pytorch import MTCNN, InceptionResnetV1
import math
import cv2
import numpy
from PIL import Image
import datetime
import torch
import os
import pandas as pd
from tkinter import *
from PIL import ImageTk, Image
from copy import deepcopy


def init_dict():
  print("Initializing")
  #test_data_dir = '/media'
  test_data_dir = './single_data_cropped'

  embedding_dict = {}

  for i,a in enumerate(os.listdir(test_data_dir)):
    print("\rInitialization Number - {}".format(i),end="")
    img_name_main = os.listdir(os.path.join(test_data_dir,a))[0]
    srcm = os.path.join(test_data_dir,a,img_name_main) 
    img = Image.open(srcm)
    img_cropped = mtcnn(img)
    img_embedding = resnet(img_cropped.unsqueeze(0).to(device))[0].detach().cpu().numpy()
    embedding_dict[a]=img_embedding

  return embedding_dict

def find_name(img_cropped,embedding_dict):

  img_cropped = Image.fromarray(img_cropped)
  img_cropped.save('x.jpg')
  img_cropped = mtcnn(img_cropped).to(device)
  img_embedding = resnet(img_cropped.unsqueeze(0))[0].detach().cpu().numpy()
  final_name = None
  print(9)
  # max = 2*math.sqrt(512)
  max = 1
  for name,emb in embedding_dict.items():
    dist = numpy.linalg.norm(img_embedding-emb)
    print("in")
    if dist<max:
      final_name = name
      max=dist
  print("\n",final_name,dist)

  return final_name,dist


def vote(name,vote_dict,threshold):

  if name not in list(vote_dict.keys()):
    vote_dict[name] = {
                        'vote':1,
                        'time': datetime.datetime.now(),
                      }
  else:
    vote_dict[name]['vote']+=1
    
  #print(vote_dict[name]['vote'])
  
  if vote_dict[name]['vote']>threshold:
    register(name,vote_dict[name]['time'])
    del vote_dict[name]

  return vote_dict

def register(name,time):

  if "register.csv" not in os.listdir():
    f = open("register.csv","w+")
    f.write("NAME,TIME")
    f.close()

  df = pd.read_csv("register.csv",index_col=None,usecols=['NAME', 'TIME'])
  df.columns = ['NAME','TIME']
  df = df.append({'':'','NAME':name,'TIME':time},ignore_index=True)
  df.to_csv("register.csv")
  # print(df)


threshold = 20
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

inmem_embeddings_dict = init_dict()
cap = cv2.VideoCapture(0)

vote_dict = {}
i = 0

font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1
color = (255, 0, 0) 
thickness = 2


if __name__=="__main__":

  root = Tk()
  app = Frame(root, bg="white")
  app.grid()
  lmain = Label(app)
  lmain.grid()



  def stream():
    
    global threshold,device,mtcnn,resnet,inmem_embeddings_dict,cap,vote_dict,i,font,fontScale,color,thickness

    try:
      _,frame = cap.read()
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      framePIL = Image.fromarray(numpy.array(frame))
      boxes, _ = mtcnn.detect(framePIL)
      frameCopy = deepcopy(frame)
      try:
        for box in boxes:
            img = frameCopy[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:]
            name,dist = find_name(img,inmem_embeddings_dict)
            vote_dict = vote(name,vote_dict,threshold)
            frame = cv2.rectangle(frame, (int(box[0]),int(box[1])),(int(box[2]),int(box[3])) , color, thickness)
            frame = cv2.putText(frame, str(name)+str(dist), (int(box[0]),int(box[1])), font,  
                      fontScale, color, thickness, cv2.LINE_AA)
      except Exception as e:
        print(e,"2")

      try:
        print('\rTracking frame: {} | Number of faces: {}'.format(i + 1,len(boxes)), end='')
      except:
        print('\rTracking frame: {} | Number of faces: {}'.format(i + 1,0), end='')
      
      # cv2.imshow("frame",frame)
      # k = cv2.waitKey(10)
      # if k==27:
      # 	break
      #result.write(frame)
      img = Image.fromarray(frame)
      imgtk = ImageTk.PhotoImage(image=img)
      lmain.imgtk = imgtk
      lmain.configure(image=imgtk)
      lmain.after(1, stream)
      
    except Exception as e:
      print(e)
      pass
    i+=1
    

  stream()
  root.mainloop()

    
cap.release()


