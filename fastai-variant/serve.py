from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.vision import load_learner,models,accuracy,ShowGraph,DataBunch,cnn_learner
import torch
import numpy as np
from pathlib import Path
from io import BytesIO
import sys
import os
import uvicorn
import aiohttp
import asyncio
import audioutils
from audioutils import AudioPreProcessor
import utils
import pickle

app=Starlette()
params=utils.load_params('./')
db=DataBunch.load_empty('saved','inferdb.pkl')
#learn=load_learner('saved','esc50-res50-model-stage2')
learn = cnn_learner(db, models.resnet50, metrics=[accuracy],callback_fns=ShowGraph, pretrained=True)
learn.load('esc50-res50-stage2')
learn.model.double()
learn.model.eval()
with open('saved/classes_to_labels.pkl','rb') as f:
    c2i=pickle.load(f)

def predict_sound_from_bytes(learn,c2i,f,params):
    preprocessor=audioutils.AudioPreProcessor('./',params)
    smi=audioutils.spectrogram(f,params)
    smit=torch.DoubleTensor(smi)
    pr=learn.model.forward(smit.reshape(1,*smit.shape))[0].detach().numpy()
    pred = {name:pr[ind] for name,ind in learn.data.c2i.items()}
    res = list(pred.items())
    s=0
    for c, val in res:
        s+=np.exp(val)
    for i in range(len(res)):
        res[i]=(res[i][0],np.exp(res[i][1])/s)
    res.sort(key=lambda x:x[1],reverse=True)
    return JSONResponse(dict(res))


@app.route("/")
def form(request):
    with open('templates/home.html','r') as f:
        st=f.read()
    return HTMLResponse(st)
@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    fileid="inputFile"
    bytes = await (data[fileid].read())
    with open(data[fileid].filename,'wb') as f:
        f.write(bytes)
    res=predict_sound_from_bytes(learn,c2i,data[fileid].filename,params)
    os.remove(data[fileid].filename)
    return res
async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

@app.route('/uploadajax', methods=['POST'])
async def upldfile(request):
    if request.method == 'POST':
        data = await request.form()
        fileid="inputFile"
        
        bytes = await (data[fileid].read())
        
        with open(data[fileid].filename,'wb') as f:
            f.write(bytes)
        res=predict_sound_from_bytes(learn,c2i,data[fileid].filename,params)
        os.remove(data[fileid].filename)
        
        return res


if __name__ == "__main__":
    
	uvicorn.run(app,host="0.0.0.0", port=80, log_level= "info",reload=True,debug=True)


