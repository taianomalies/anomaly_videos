from google.colab import drive
drive.mount('/content/drive')

optical_flow_dataset, height, width, channels, frames= calculate_optical_flow_datasets(['/content/drive/MyDrive/datasets/']) #sciezka do podfolderow zawierajacych film juz podzielony na klatki 
#optical_flow_dataset = np.reshape(optical_flow_dataset, (optical_flow_dataset.shape[1], 255, 255, 2))
optical_flow_dataset = np.reshape(optical_flow_dataset, (optical_flow_dataset.shape[1], optical_flow_dataset.shape[2], optical_flow_dataset.shape[3], 2))#

num_frames = optical_flow_dataset.shape[0]
window_size = 8
# Create windows
windows = []
for i in range(num_frames - window_size + 1):
    window = optical_flow_dataset[i : i + window_size]
    windows.append(window)

# Convert the list of windows to a NumPy array
windows = np.array(windows)
windows.shape

autoencoder = create_AE(windows)
batch_size = 1
num_epochs = 3
trained_autoencoder = train_model(autoencoder, windows, batch_size, num_epochs)
