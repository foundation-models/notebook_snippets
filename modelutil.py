import keras
from keras.models import model_from_json
from keras.models import model_from_yaml

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

def load_model_json(dir, model_name):
  return load_model(dir, model_name, '.json')
def load_model_yaml(dir, model_name):
  return load_model(dir, model_name, '.yaml')
def load_model(dir, model_name, extension):
  # load json and create model
  file_name = dir + '/model.' + model_name + extension
  file = open(file_name, 'r')
  loaded_model = file.read()
  file.close()
  print('load model from file ' + file_name)
  if(extension == '.json'):
    return model_from_json(loaded_model)
  if(extension == '.yaml'):
    return model_from_yaml(loaded_model)
  return 'no valid extension'

def save_model_json(dir, model, model_name):
  save_model(dir, model, model_name, '.json')
def save_model_yaml(dir, model, model_name):
  save_model(dir, model, model_name, '.yaml')
def save_model(dir, model, model_name, extension):
  # serialize model to JSON
  file_name = dir + '/model.' + model_name + extension
  if(extension == '.json'):
    model_loaded = model.to_json()
  if(extension == '.yaml'):
    model_loaded = model.to_yaml()
  with open(file_name, "w") as file:
      file.write(model_loaded)
  save_model_weights(dir, model, model_name)
def save_model_weights(dir, model, model_name):
  # serialize weights to HDF5
  file_name = dir + '/model.' + model_name + '.h5'
  model.save_weights(file_name)
  
def load_model_weights(dir, model, model_name):
  # load serialize weights from HDF5
  file_name = dir + '/model.' + model_name + '.h5'
  model.load_weights(file_name)
  
print('save and load models from yaml and json files defined.\
 Everything stored in folder ', dir)
