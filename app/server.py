import aiohttp
import asyncio
import uvicorn
#import cv2
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles


#export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
#export_file_name = 'export.pkl'

#classes = ['black', 'grizzly', 'teddys']

#export_file_url = 'https://drive.google.com/uc?export=download&id=1N9lWOHSMaxN322XRyAALovNxCLzM6NCT' #'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
#export_file_name = 'rash1.pkl'

#export_file_url1 = 'https://drive.google.com/uc?export=download&id=1CiGOOrzrEGF0UCljBtkZl8H5aK1VNNZn'
#export_file_name1 = 'rash2.pkl'

export_file_url1 = 'https://drive.google.com/uc?export=download&id=1HlEDUK5n5yHxnnFS5G19DEqnTuOyFBYT'
export_file_name1 = 'rash1_new100.pkl'

#export_file_url1 = 'https://drive.google.com/uc?export=download&id=1qAXSZplAldJIrDnE_NryZJmpx3kOYZIo'
#export_file_name1 = 'rash152res200.pkl'

#export_file_url = 'https://drive.google.com/uc?export=download&id=1YaGpjyiNNEGzzG5xzwzsVnYJSwIrqyWz'
#export_file_name = 'rash_152_200.pkl'

#Melanoma and Normal Rash seperate decision

#export_file_url1= 'https://drive.google.com/uc?export=download&id=1zUdnIy8bm8zOReVqt0VMi234o6ryREIe'
#export_file_name1= 'normalrash.pkl'

#export_file_url2 = 'https://drive.google.com/uc?export=download&id=1j2HpbwTb078ykjQPd2gIybcx0jb4G05R'
#export_file_name2= 'melanoma.pkl'

#classes = ['Flea Bites', 'Tick Bites']
#classes = ['Allergic Eczema', 'Cellulitis', 'Chickenpox', 'Contact Dermatitis', 'Diaper Rash', 'Drug Allergy', 'Eczema' ,'Fifth Disease', 'Flea Bites','Impetigo','Kawasaki Disease','Measles','Psoriasis','Ringworm', 'Rosacea','Scabies','Scarle Fever','Sebornheic Eczema','Shingles','Systemic Lupus Erythematosus','Tick Bites']
classes = ['Allergic Eczema', 'Cellulitis', 'Chickenpox', 'Contact Dermatitis', 'Drug Allergy', 'Eczema', 'Flea bite', 'HFMD', 'Impetigo', 'Measles', 'Melanoma', 'Psoriasis', 'Ringworm', 'Rosacea', 'SLE', 'Scabies', 'Scarlet Fever', 'Seborrhoeic Eczema', 'Shingles', 'Tick Bite']
path = Path(__file__).parent

#Starlette app Globals
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url1, path / export_file_name1)

    try:
        learn1 = load_learner(path, export_file_name1)
        
        return learn1
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise
           
'''
async def setup_learner2():
    await download_file(export_file_url2, path / export_file_name2)
    try:

        learn2 = load_learner(path, export_file_name2)
        return learn2
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise
'''            
            
loop = asyncio.get_event_loop()
tasks1 = [asyncio.ensure_future(setup_learner())]
#tasks2 = [asyncio.ensure_future(setup_learner2())]
learn1 = loop.run_until_complete(asyncio.gather(*tasks1))[0]
#learn2 = loop.run_until_complete(asyncio.gather(*tasks2))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    #check if the image is skin image
    
    #check if the image blurred
    #fm=cv2.Laplacian(img,cv2.CV_64F).var()
    #if fm<125:
     #   return JSONResponse({'result':'Sorry, please upload a clearer image.'})
    
    #Melanoma detect
    #ans2= learn2.predict(img)
    #cancer= ans2[0]
    #p=ans2[2]
    ans1 = learn1.predict(img)
    prediction = ans1[0]
    probability = ans1[2]
    #return rash names, probabilities, and treatment
    
    return JSONResponse({'result': str(prediction),'probability':str(probability)})
    #return JSONResponse({'melanoma':str(cancer),'m_probability':str(p),'result':str(prediction),'probability':str(probability)})

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
