from torchvision import models 
import torch     
dir(models)

resnet = models.resnet101(pretrained = True)
print (resnet)

from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

from zipfile import ZipFile
filename1 = 'E:\Study\BK\HK231\TGM\BTAP\HW7/black_grouse.zip'
filename2 = 'E:\Study\BK\HK231\TGM\BTAP\HW7/brambling.zip'
filename3 = 'E:\Study\BK\HK231\TGM\BTAP\HW7\koala.zip'
filename4 = 'E:\Study\BK\HK231\TGM\BTAP\HW7\ostrich.zip'
filename5 = 'E:\Study\BK\HK231\TGM\BTAP\HW7/tree_frog.zip'
with ZipFile(filename1, 'r') as zip:
  zip.extractall()
with ZipFile(filename2, 'r') as zip:
  zip.extractall()
with ZipFile(filename3, 'r') as zip:
  zip.extractall()
with ZipFile(filename4, 'r') as zip:
  zip.extractall()
with ZipFile(filename5, 'r') as zip:
  zip.extractall()

print ("Done!")

from torch.nn.modules.fold import Fold
import os
import cv2
FolderPath1 = 'E:\Study\BK\HK231\TGM\BTAP\HW7/black_grouse'
myList1 = os.listdir(FolderPath1)
#overlayList = []
count_T_class1 = 0
count_F_class1 = 0
count_T5_class1 = 0
count_F5_class1 = 0

for imPath1 in myList1:
  img = cv2.imread(f'{FolderPath1}/{myList1}')

  from PIL import Image 
  Path1 = 'E:\Study\BK\HK231\TGM\BTAP\HW7/black_grouse/' + imPath1
  anh1 = Image.open(Path1)
  img_t1 = transform(anh1)
  batch_t = torch.unsqueeze(img_t1, 0)
  resnet.eval()
  out = resnet(batch_t)
  out.shape
  with open("E:\Study\BK\HK231\TGM\BTAP\HW7/imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]
  _, index = torch.max(out, 1)
  percentage = torch.nn.functional.softmax(out, dim=1)[0]
  
  print([(classes[index], percentage[index].item())]) 
  if (index == 1):
    count_T_class1 += 1
  else:
    count_F_class1 += 1

    
  _, indices = torch.sort(out, descending = True)
  percentage1 = torch.nn.functional.softmax(out, dim=1)[0]
  print([(classes[idx], percentage1[idx].item()) for idx in indices[0][:5]])
  limit = (indices[0][:5])
  if (limit[0]==1 or limit[1]==1 or limit[2]==1 or limit[3]==1 or limit[4]==1):
    count_T5_class1 += 1
  else:
    count_F5_class1 += 1

print ("Count_F_class1: " + str(count_F_class1))
print ("Count_F5_class1 " + str(count_F5_class1))


from torch.nn.modules.fold import Fold
import os
import cv2
FolderPath2 = 'E:\Study\BK\HK231\TGM\BTAP\HW7/brambling'
myList2 = os.listdir(FolderPath2)
#overlayList = []
count_T_class2 = 0
count_F_class2 = 0
count_T5_class2 = 0
count_F5_class2 = 0

for imPath2 in myList2:
  img = cv2.imread(f'{FolderPath2}/{myList2}')
  
  from PIL import Image 
  Path2 = 'E:\Study\BK\HK231\TGM\BTAP\HW7/brambling/' + imPath2
  anh2 = Image.open(Path2)
  img_t2 = transform(anh2)
  batch_t = torch.unsqueeze(img_t2, 0)
  resnet.eval()
  out = resnet(batch_t)
  out.shape
  with open("E:\Study\BK\HK231\TGM\BTAP\HW7/imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]
  _, index = torch.max(out, 1)
  percentage = torch.nn.functional.softmax(out, dim=1)[0]
  
  print([(classes[index], percentage[index].item())]) 
  if (index == 8):
    count_T_class2 += 1
  else:
    count_F_class2 += 1

    
  _, indices = torch.sort(out, descending = True)
  percentage1 = torch.nn.functional.softmax(out, dim=1)[0]
  print([(classes[idx], percentage1[idx].item()) for idx in indices[0][:5]])
  limit = (indices[0][:5])
  if (limit[0]==8 or limit[1]==8 or limit[2]==8 or limit[3]==8 or limit[4]==8):
    count_T5_class2 += 1
  else:
    count_F5_class2 += 1

print ("Count_F_class2 " + str(count_F_class2))
print ("Count_F5_class2 " + str(count_F5_class2))


from torch.nn.modules.fold import Fold
import os
import cv2
FolderPath3 = 'E:\Study\BK\HK231\TGM\BTAP\HW7\koala'
myList3 = os.listdir(FolderPath3)
#overlayList = []
count_T_class3 = 0
count_F_class3 = 0
count_T5_class3 = 0
count_F5_class3 = 0

for imPath3 in myList3:
  img = cv2.imread(f'{FolderPath3}/{myList3}')

  from PIL import Image 
  Path3 = 'E:\Study\BK\HK231\TGM\BTAP\HW7\koala/' + imPath3
  anh3 = Image.open(Path3).convert('RGB')
  #print (Path3)
  
  img_t3 = transform(anh3)
  batch_t = torch.unsqueeze(img_t3, 0)
  resnet.eval()
  out = resnet(batch_t)
  out.shape
  with open("E:\Study\BK\HK231\TGM\BTAP\HW7/imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]
  _, index = torch.max(out, 1)
  percentage = torch.nn.functional.softmax(out, dim=1)[0]
  
  print([(classes[index], percentage[index].item())]) 
  if (index == 21):
    count_T_class3 += 1
  else:
    count_F_class3 += 1

    
  _, indices = torch.sort(out, descending = True)
  percentage1 = torch.nn.functional.softmax(out, dim=1)[0]
  print([(classes[idx], percentage1[idx].item()) for idx in indices[0][:5]])
  limit = (indices[0][:5])
  if (limit[0]==21 or limit[1]==21 or limit[2]==21 or limit[3]==21 or limit[4]==21):
    count_T5_class3 += 1
  else:
    count_F5_class3 += 1

print ("Count_F_class3 " + str(count_F_class3))
print ("Count_F5_class3 " + str(count_F5_class3)) 


from torch.nn.modules.fold import Fold
import os
import cv2
FolderPath4 = 'E:\Study\BK\HK231\TGM\BTAP\HW7\ostrich'
myList4 = os.listdir(FolderPath4)
#overlayList = []
count_T_class4 = 0
count_F_class4 = 0
count_T5_class4 = 0
count_F5_class4 = 0

for imPath4 in myList4:
  img = cv2.imread(f'{FolderPath4}/{myList4}')
  
  from PIL import Image 
  Path4 = 'E:\Study\BK\HK231\TGM\BTAP\HW7\ostrich/' + imPath4
  anh4 = Image.open(Path4)
  img_t4 = transform(anh4)
  batch_t = torch.unsqueeze(img_t4, 0)
  resnet.eval()
  out = resnet(batch_t)
  out.shape
  with open("E:\Study\BK\HK231\TGM\BTAP\HW7/imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]
  _, index = torch.max(out, 1)
  percentage = torch.nn.functional.softmax(out, dim=1)[0]
  
  print([(classes[index], percentage[index].item())]) 
  if (index == 9):
    count_T_class4 += 1
  else:
    count_F_class4 += 1

    
  _, indices = torch.sort(out, descending = True)
  percentage1 = torch.nn.functional.softmax(out, dim=1)[0]
  print([(classes[idx], percentage1[idx].item()) for idx in indices[0][:5]])
  limit = (indices[0][:5])
  if (limit[0]==9 or limit[1]==9 or limit[2]==9 or limit[3]==9 or limit[4]==9):
    count_T5_class4 += 1
  else:
    count_F5_class4 += 1

print ("Count_F_class4 " + str(count_F_class4))
print ("Count_F5_class4 " + str(count_F5_class4))


from torch.nn.modules.fold import Fold
import os
import cv2
FolderPath5 = 'E:\Study\BK\HK231\TGM\BTAP\HW7/tree_frog'
myList5 = os.listdir(FolderPath5)
#overlayList = []
count_T_class5 = 0
count_F_class5 = 0
count_T5_class5 = 0
count_F5_class5 = 0

for imPath5 in myList5:
  img = cv2.imread(f'{FolderPath5}/{myList5}')

  from PIL import Image 
  Path5 = 'E:\Study\BK\HK231\TGM\BTAP\HW7/tree_frog/' + imPath5
  anh5 = Image.open(Path5)
  img_t5 = transform(anh5)
  batch_t = torch.unsqueeze(img_t5, 0)
  resnet.eval()
  out = resnet(batch_t)
  out.shape
  with open("E:\Study\BK\HK231\TGM\BTAP\HW7/imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]
  _, index = torch.max(out, 1)
  percentage = torch.nn.functional.softmax(out, dim=1)[0]
  
  print([(classes[index], percentage[index].item())]) 
  if (index == 2):
    count_T_class5 += 1
  else:
    count_F_class5 += 1

    
  _, indices = torch.sort(out, descending = True)
  percentage1 = torch.nn.functional.softmax(out, dim=1)[0]
  print([(classes[idx], percentage1[idx].item()) for idx in indices[0][:5]])
  limit = (indices[0][:5])
  if (limit[0]==2 or limit[1]==2 or limit[2]==2 or limit[3]==2 or limit[4]==2):
    count_T5_class5 += 1
  else:
    count_F5_class5 += 1

print ("Count_F_class5 " + str(count_F_class5))
print ("Count_F5_class5 " + str(count_F5_class5))


total_F = count_F_class1 + count_F_class2 + count_F_class3 + count_F_class4 + count_F_class5
total_F5 = count_F5_class1 + count_F5_class2 + count_F5_class3 + count_F5_class4 + count_F5_class5

top1 = (total_F/100)*100
top5 = (total_F5/100)*100

print ("Top 1 error rate: " + str(top1) + "%")
print ("Top 5 error rate: " + str(top5) + "%")