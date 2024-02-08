import numpy as np
import matplotlib.pyplot as plt
import os
# Load the .npy file
print(os.getcwd())
batches = np.load('src/Vision/testarr.npy')
# batches = np.load('src/Vision/testarr.npy')


# Select which batch to plot, for example, the first batch
selected_batch_index = 0
selected_batch = batches

# Plotting
fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # Assuming 10 images per batch, arrange them in 2 rows and 5 columns
fig.suptitle(f'Images from batch {selected_batch_index+1}')

# Add index labels
for i, ax in enumerate(axs.flat):
    ax.imshow(selected_batch[i, 1, :, :])  
    ax.axis('off')  # Hide axes for better visualization
    ax.set_title(f'Image {i}')  # Add index label to each image

plt.savefig('src/Vision/one.png')  # Show the plot
plt.show()  # Save the plot
