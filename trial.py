import pickle
import tensorflow as tf
model_dir = "./cnn_model"
loaded_model1 = tf.keras.models.load_model(model_dir)
pred=[[1,1,1,1,1,1,1]]
predicted = loaded_model1.predict(pred)
prediction = list(predicted[0])
print(prediction.index(max(prediction)))