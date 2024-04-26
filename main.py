import uvicorn
from fastapi import FastAPI, UploadFile, File
from keras.models import load_model
from predict import predict_digit
from io import BytesIO


import numpy as np
from PIL import Image
#load model
model=load_model("mnist_model.h5")
#create fastapi app
app = FastAPI()

#format image
def format_image(image):
    #resize image into (28, 28) and then into gray image
    image = image.resize((28, 28)).convert('L')
    #reshape into 1d 784 value array
    image_array = np.array(image).reshape(784)
    return image_array


@app.post('/predict')
async def predict(upload_file: UploadFile = File(...)):
    #read inpput file
    file = await upload_file.read()
    image = Image.open(BytesIO(file))
    #format image
    img_array = format_image(image)
    print(img_array.max())
    # predict image
    digit = predict_digit(model, img_array)
    return {"digit": digit}