import pandas as pd
import numpy as np
import clip
from PIL import Image
import torch

from torch.utils.data import DataLoader,Dataset


device="cuda"
model_name="RN50"
global model,preprocess

class ImageData(Dataset):
    def __init__(self,imagepath_list,labels_list):
        super(self,ImageData).__init__()
        self.name=imagepath_list
        self.label=labels_list
    
    def __getitem__(self, index):
        imagepath=self.name[index]
        label=self.label[index]
        image=preprocess(Image.open(imagepath)).unsqueeze(0)
        return image,label
        

class FairnessMetric():
    def __init__(self):
        # self.device="cuda"
        # model_name="RN50"
        # self.model,self.preprocess=clip.load(model_name,device=self.device)
        self.sensitive_words=["a female","a male","no human face"]
        self.text=clip.tokenize(self.sensitive_words).to(self.device)
        
    def set_sensitive_attribute(self,attribute_list):
        self.sensitive_words=attribute_list
        
    def predict_attribute(self,image):
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, self.text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs
        
    # def gender_classification(self,image_path,keyword=""):
        
    #     image=self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
    #     print(image)
    #     with torch.no_grad():
    #         logits_per_image, logits_per_text = self.model(image, self.text)
    #         probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
    #     print(probs)
    
fair=FairnessMetric()
model,preprocess=clip.load(model_name,device=device)
race_attributes=["an Asian","a Black", "an Indian", "a White","no human face"]
# fair.gender_classification("/home/codes/diffusion/images/race-occupation-layer8-0.1/author/00003.png")
# fair.gender_classification("/home/codes/diffusion/images/race-occupation-layer8-0.1/accountant/00000.png")
csvpath=""
csvfile=pd.read_csv(csvpath)
imagelist=csvfile["image_path"]
labellist=csvfile["occupations"]

ds=ImageData(imagelist,labellist)
results=[]
dl=DataLoader(ds,32)
for image,label in dl:
    score=fair.predict_attribute(image)
    pre_id=np.argmax(score,axis=1)
    attributes=fair.sensitive_words[pre_id]
    results=results+attributes
    
df=pd.DataFrame({
    "label":labellist,
    "attribute":results
})

df.to_csv("fairresult.csv",index=None)

# fair counts: