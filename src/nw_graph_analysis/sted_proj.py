import imageio
import numpy as np

for seq in range(1, 17):
    try:
        proj = np.zeros((128, 128))
        for i in range(100):
            img = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/Climp/std/C{seq}_decon_t0{i:02d}_ch00_std.png')
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            proj += img
        imageio.imwrite(f'/localhome/asa420/MIAL/data/sted-data/Climp/climp{seq}_er_mean.png', proj)
    except Exception:
        pass


dl = DataLoader(dataset, batch_size=4, shuffle=True)

# visualize the data

