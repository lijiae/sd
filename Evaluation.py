import pandas
import argparse
from metric.datareader import *
from metric.MetricTool import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import PIL

def makeargs():
    args=argparse.ArgumentParser()
    args.add_argument("--result_csv",type=str,default="/ibex/user/lij0w/codes/data/reuslt_csv/orign.csv")
    args.add_argument("--bz",type=int,default=32)
    opt = args.parse_args()
    return opt

args=makeargs()
dataset=ImageDataset(args.result_csv,device='cpu')
dl=DataLoader(dataset,args.bz)
promt="an image of a {}"

# consis=Consistency()

sum=0
for image,label in tqdm(dl):
    # score=consis.model(image,label)
    image /= image.norm(dim=-1, keepdim=True)
    label /= label.norm(dim=-1, keepdim=True)
    # score = (image @ label.T).softmax(dim=-1)
    score = (image @ label.T)
    # print(score)
    score=torch.diag(score)
    # print(score)
    sum+=score.sum().item()
print(sum)
print(len(dataset))
print(sum/len(dataset))
