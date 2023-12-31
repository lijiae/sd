import pandas as pd
import numpy as np
import clip
from PIL import Image
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader,Dataset


device="cuda"
model_name="RN50"
global model,preprocess

class ImageData(Dataset):
    def __init__(self,imagepath_list,labels_list):
        super(ImageData,self).__init__()
        self.name=imagepath_list
        self.label=labels_list
        
    def __len__(self):
        return len(self.name)
    
    def __getitem__(self, index):
        imagepath=self.name[index]
        label=self.label[index]
        image=preprocess(Image.open(imagepath))
        return image,label
        

class FairnessMetric():
    def __init__(self):
        self.sensitive_words=["a female","a male","no human face"]
        self.text=clip.tokenize(self.sensitive_words).to(device)
        
    def set_sensitive_attribute(self,attribute_list):
        self.sensitive_words=attribute_list
        
    def predict_attribute(self,image):
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, self.text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs
        
    
fair=FairnessMetric()
model,preprocess=clip.load(model_name,device=device)
race_attributes=["an Asian face","a Black face", "an Indian face", "a White man face","no human face"]
# fair.set_sensitive_attribute(race_attributes)
csvpath="/ibex/user/lij0w/codes/data/reuslt_csv/orign.csv"
csvfile=pd.read_csv(csvpath)
imagelist=csvfile["image_path"]
labellist=csvfile["label"]

ds=ImageData(imagelist,labellist)
results=[]
dl=DataLoader(ds,32)
for image,label in tqdm(dl):
    score=fair.predict_attribute(image.to(device))
    pre_id=np.argmax(score,axis=1)
    attributes=[fair.sensitive_words[i] for i in pre_id]
    results=results+attributes
    
df=pd.DataFrame({
    "label":labellist,
    "attribute":results
})

df.to_csv("gender_fairresult.csv",index=None)

# fair counts: