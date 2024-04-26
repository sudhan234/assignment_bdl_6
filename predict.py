import numpy as np

def predict_digit(model,data):
    """
    predicts the digit given the data in the form of a matrix
    """ 
    data = np.array(data)
    #normalize the data
    data=data/max(data)
    #predict the output matrix
    output = model.predict(data.reshape(1,-1))
    #argmax of output
    digit = np.argmax(output)
    return str(digit)