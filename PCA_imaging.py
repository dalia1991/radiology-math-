import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Create a 10x10 "Signal" (e.g., a diagonal gradient like a scan feature)
x = np.linspace(0, 255, 10)
X_grid, Y_grid = np.meshgrid(x, x)
signal = (X_grid + Y_grid) / 2

# 2. Add "Medical Sensor Noise"
np.random.seed(42)
noise = np.random.normal(0, 50, (10, 10)) # Heavy noise
noisy_image = np.clip(signal + noise, 0, 255)

# 3. PCA Reconstruction at different levels of "Information"
# We'll try k=1 (Main trend) and k=3 (More detail)
def reconstruct_pca(data, k):
    pca = PCA(n_components=k)
    # Fit and transform the columns
    components = pca.fit_transform(data)
    # Reconstruct back to 10x10
    return pca.inverse_transform(components)

rec_k1 = reconstruct_pca(noisy_image, k=1)
rec_k3 = reconstruct_pca(noisy_image, k=3)

# 4. Visualization
images = [signal, noisy_image, rec_k1, rec_k3]
titles = ["True Signal", "Noisy Input", "PCA k=1 (Extreme)", "PCA k=3 (Balanced)"]

plt.figure(figsize=(12, 4))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
