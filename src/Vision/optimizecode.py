import tensorflow as tf

new_model = tf.keras.models.load_model('./src/Vision/action.keras',safe_mode=False)

print(new_model.summary())

img_path = './read10_rgb_2_frame31.png'
img = load_img(img_path, target_size=(128, 128))  # assuming you need the image to be 128x128
img = img_to_array(img)
img = np.expand_dims(img, axis=0)  # model.predict expects a batch of images

predicted  = model.predict(img,batch_size = 10)
print(predicted[0])