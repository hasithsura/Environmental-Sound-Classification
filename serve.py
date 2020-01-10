from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import torch
import numpy as np
from io import BytesIO
import sys
import os
import gunicorn
import uvicorn
import aiohttp
import asyncio
import pickle
from audioutils import get_melspectrogram_db, spec_to_image

app=Starlette()
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
with open('saved/indtocat.pkl','rb') as f:
    indtocat = pickle.load(f)
resnet_model = torch.load('saved/esc50resnet.pth', map_location=device)
def predict_sound_from_bytes(resnet_model,indtocat, filename):
    spec=get_melspectrogram_db(filename)
    spec_norm = spec_to_image(spec)
    spec_t=torch.tensor(spec_norm).to(device, dtype=torch.float32)
    pr=resnet_model.forward(spec_t.reshape(1,1,*spec_t.shape))[0].cpu().detach().numpy()
    pred = {name:pr[ind] for ind,name in indtocat.items()}
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
    res=predict_sound_from_bytes(resnet_model,indtocat,data[fileid].filename)
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
        res=predict_sound_from_bytes(resnet_model,indtocat,data[fileid].filename)
        os.remove(data[fileid].filename)
        
        return res


if __name__ == "__main__":    
	uvicorn.run(app,host="0.0.0.0", port=5000, log_level= "info",debug=True)

