import numpy as np

def segment_table(rgb: np.ndarray):
    threshold = 0.55
    proportions = rgb / np.sum(rgb, axis=2)[...,None]
    blue_dominant = proportions[:,:,2] > threshold
    return (rgb * blue_dominant[...,None])[:,:,[2,1,0]]
    print(proportions[:,:,2])
    print(proportions[:,:,2] > threshold)
    print((rgb * blue_dominant[...,None])[:,:,2])
    return rgb * blue_dominant[...,None]
    # return proportions