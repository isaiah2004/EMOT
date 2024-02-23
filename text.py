import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
else:
    print("CUDA is not available. Defaulting to CPU.")


# import json
# from tensorflow.keras.models import model_from_json

# # Load the original JSON file
# with open('model.json', 'r') as file:
#     data = json.load(file)

# # Extract the model topology part from the TensorFlow.js format
# model_topology = data['modelTopology']

# # Convert the model topology back to JSON string
# model_json = json.dumps(model_topology)

# # Load the model from JSON
# model = model_from_json(model_json)

# # Load weights into the model (adjust the path as necessary)
# model.load_weights('weights.bin')

# # Save the model and weights in HDF5 format
# model.save('model.h5')
