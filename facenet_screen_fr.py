import os
from os import listdir
import numpy as np
import cv2

from facenet_pytorch import MTCNN, InceptionResnetV1
import math

import torch

import h5py
import cv2
import numpy as np
import os
import glob

import numpy as np
import cv2
from mss import mss
from PIL import Image


import pygame
import win32api
import win32con
import win32gui
from ctypes import windll
import time
import threading

workingimgdir = 'images'





def display(loc):
    global SetWindowPos,screen,white,green,blue,font,hwnd,fuchsia,dark_red

    screen.fill(fuchsia)
    for name,xmin,ymin,xmax,ymax in loc:
        text = font.render(name, True, green)
        pygame.draw.rect(screen, dark_red, pygame.Rect(xmin, ymin, xmax, ymax),2)
        textRect = text.get_rect()
        textRect.center = ((2*xmin+xmax)//2, ymin-17)
        screen.blit(text, textRect)
    pygame.display.update()
    # time.sleep(3)



mon = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
sct = mss()

def get_frame():
    global sct
    sct.get_pixels(mon)
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    img = im_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img,(1920,1080))
    return img





class HDF5Store(object):
    def __init__(self, datapath, dataset, shape=(1,), dtype=np.float32, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.vecdim = 512
        #Special 
        dtype = np.dtype([('Name', 'S32'), ('Vector', np.float32, (self.vecdim,)),('Valid','i')])
        self.shape = (1,)
        self.dtype = dtype
        self.inh5 = set()
        self.min_name = None
        self.min_dist = 1000
        self.g = 0



        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))
        self.mtcnn = MTCNN(image_size=160, margin=0, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()








        if self.datapath not in list(os.listdir()):

            with h5py.File(self.datapath, mode='w') as h5f:
                self.dset = h5f.create_dataset(
                    dataset,
                    shape=(0, ) + shape,
                    maxshape=(None, ) + shape,
                    dtype=dtype,
                    compression=compression,
                    chunks=(chunk_len, ) + shape)
                h5f.flush()
                h5f.close()
            self.i = 0
        else:
            with h5py.File(self.datapath, mode='r') as h5f:
                self.i = h5f[dataset].shape[0]
                print('/////////',self.i)
                self.inh5 = set(map(lambda x:x.decode("utf-8") ,h5f[dataset]['Name'].flatten()))
                h5f.close()
                
        self.checknewold()

        with h5py.File(self.datapath, mode='r') as h5f:
            self.veclib = np.array(list(map(self.l2_normalize,h5f[self.dataset]['Vector'].reshape(-1,self.vecdim))))
            self.lendataset = len(self.veclib)
            print('List')
            for x,y in zip(h5f['vecs']['Name'],h5f['vecs']['Valid']):
                print(x[0],y[0])
            print("-------------")



    def checknewold(self):
        unfound_face_list = []
        listdir = os.listdir('images')
        setlistdir = set(listdir)
        inh5 = self.inh5

        tobeadded = setlistdir-inh5
        toberemoved = inh5 - setlistdir - {'Unknown'}
        print("tobeadded ",tobeadded)
        print("toberemoved",toberemoved)
        for i in tobeadded:
            try: 
                print("\rAdding: {}".format(i),end='')

                # try:
                #     with h5py.File(self.datapath, mode='r+') as h5f:
                #         if len(h5f['vecs'][h5f['vecs']['Name']==i,'Valid'])==1:
                #             h5f[self.dataset][h5f[self.dataset]['Name']==i,'Valid'] = np.array([1],dtype=np.int32)
                #             continue
                # except Exception as e:
                #     print(e)
                #     pass

                fin_path = os.path.join('images',i)
                
                img = Image.open(fin_path)
                img_cropped = self.mtcnn(img)

                representation = self.resnet(img_cropped.unsqueeze(0).to(self.device))[0].detach().cpu().numpy()
                    
            except Exception as e:
                print(e)
                unfound_face_list.append(i)
                continue



            self.append(np.array([(i,representation,1)],dtype=self.dtype))

        print("\n\nCouldnt find faces in the following images")
        for x in unfound_face_list:
            print("\t{}".format(x))

        for i in toberemoved:
            print("\rRemoving: {}".format(i),end='')
            self.remove(i)

        del listdir
        del unfound_face_list
        del inh5
        del setlistdir
        del tobeadded
        del toberemoved

    def append(self, values):
        with h5py.File(self.datapath, mode='a') as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + 1, ) + self.shape)
            dset[self.i] = [values]
            self.i += 1
            # print(values[0][0])
            # print(h5f[self.dataset][h5f[self.dataset]['Name'][:]==values[0][0],0,'Valid'])
            h5f.flush()
            h5f.close()
            
    def remove(self, name):
        print('/////////',name)
        with h5py.File(self.datapath, mode='r+') as h5f:
            h5f[self.dataset][h5f[self.dataset]['Name']==name.encode('UTF-8'),'Valid'] = np.array([0],dtype=np.int32)
            h5f[self.dataset][h5f[self.dataset]['Name']==name.encode('UTF-8'),'Name'] = np.array([b'Unknown'],dtype='|S32')
            # h5f[][h5f[self.dataset]['Name'][:]==name,0,'Valid'] = 0    
            h5f.flush()
            h5f.close()


    def l2_normalize(self,x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))

    def findEuclideanDistance(self,source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance


    def findnearest(self,capvec):
        name = None
        dist = 1.5
        with h5py.File(self.datapath, mode='r') as h5f:
            for i in h5f[self.dataset]:
                i = i[0]
                ittervec = i[1]
                distc = self.findEuclideanDistance(self.l2_normalize(capvec), self.l2_normalize(ittervec))
                if dist>distc and i[2]==1:
                    dist = distc
                    name = i[0]

        return name.decode("utf-8"),dist

    def findnearest2(self,capvec):
        fin = np.linalg.norm(self.veclib-self.l2_normalize(capvec),axis=1)
        min = np.argmin(fin)

        if fin[min]<1:
            with h5py.File(self.datapath, mode='r') as h5f:
                name = h5f['vecs']['Name'][min][0]
            return name.decode("utf-8"),fin[min]
        else:
            return None,0

    def setminnamedist(self,name,dist,val):
        self.min_name = name
        self.min_dist = dist
        self.min_val = val

    def findnearestt(self,capvec,start,end):
        fin = np.linalg.norm(self.veclib[start:end]-self.l2_normalize(capvec),axis=1)
        min = np.argmin(fin)
        # print(fin)
        if fin[min]<self.min_dist:
            with h5py.File(self.datapath, mode='r') as h5f:
                name = h5f['vecs']['Name'][start+min][0]
                val = h5f['vecs']['Valid'][start+min][0]#h5f[self.dataset][h5f[self.dataset]['Name'][:]==name,0,'Valid']
            self.setminnamedist(name.decode("utf-8"),fin[min],val)

    def multithreadedsearch(self,capvec):

        self.g+=1
        # print(self.g)


        threadlist = []
        x = self.lendataset
        i=0
        for i in range(min(100,x//100)):
            t = threading.Thread(target=self.findnearestt, args=(capvec,x//100*i,x//100*(i+1)))
            
            threadlist.append(t)


        t = threading.Thread(target=self.findnearestt, args=(capvec,x//100*(i+1),x-1))

        threadlist.append(t)
        

        for t in threadlist:
            t.start()
        for t in threadlist:
            t.join()

        min_name,min_dist = self.min_name,self.min_dist

        self.min_name,self.min_dist = None,1000




        if min_dist<1 and self.min_val:
            return min_name,min_dist
        else:
            return None,0            


    def getframedetails(self,frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_cropped = Image.fromarray(frame)
        # img_cropped = self.mtcnn(img_cropped).to(self.device)
        # img_cropped = torch.Tensor(img_cropped).to(self.device)
        img_cropped = self.mtcnn(img_cropped).to(self.device)
        img_embedding = self.resnet(img_cropped.unsqueeze(0))[0].detach().cpu().numpy()
        name,dist = self.multithreadedsearch(capvec=img_embedding)

        return name,dist




vech5 = HDF5Store('embeddingVec.h5','vecs',)






SetWindowPos = windll.user32.SetWindowPos

pygame.init()
screen = pygame.display.set_mode((1920, 1080))
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
font = pygame.font.Font('freesansbold.ttf', 32)

fuchsia = (255, 0, 128)  
dark_red = (139, 0, 0)

hwnd = pygame.display.get_wm_info()["window"]
win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                       win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)

win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*fuchsia), 0, win32con.LWA_COLORKEY)

SetWindowPos(hwnd,-1,1920,1080,0,0,0x0003)











acc = [[".",0,0,1,1]]
display(acc)
for event in pygame.event.get(): 
        if event.type == pygame.quit:
            pygame.quit()











while(True):
    img = get_frame()

    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    framePIL = Image.fromarray(np.array(frame))
    faces, _ = vech5.mtcnn.detect(framePIL)
    try:
        if faces == None:
            continue
    except:
        pass

    acc = []


    for (xmin,ymin,xmax,ymax) in faces:
        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)


        w = xmax-xmin
        h = ymax-ymin
        x = xmin
        y = ymin
        if w > 130: #discard small detected faces
            cv2.rectangle(img, (x,y), (x+w,y+h), (67, 67, 67), 1) #draw rectangle to main image
            
            detected_face = np.array(img[int(y):int(y+h), int(x):int(x+w)]) #crop detected face
            # detected_face = cv2.resize(detected_face, target_size) #resize to 152x152
            
            # img_pixels = image.img_to_array(detected_face)
            # img_pixels = np.expand_dims(img_pixels, axis = 0)
            # img_pixels /= 255
            


            employee_name = None
            similarity = 0

            
            try:
                employee_name,similarity = vech5.getframedetails(detected_face)

            except:
                pass





            
            if employee_name!=None:
                acc.append([employee_name.split(".")[0],x,y,w,h])

            if employee_name==None:
                acc.append(["Unknown",x,y,w,h])
    
    
    display(acc)

    for event in pygame.event.get(): 
        if event.type == pygame.quit:
            pygame.quit()
    # cv2.imshow('img',img)
    
    # if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
    #     break
    
#kill open cv things        
cv2.destroyAllWindows()
