from scipy.ndimage import uniform_filter, gaussian_gradient_magnitude
import numpy as np


class GraynessIndex:
    def __init__(
        self,
        percentage_of_GPs: float = 0.1,
        delta_threshold: float = 1e-4,
        epsilon: float = 1e-7
    ) -> None:
        self.percentage_of_GPs = percentage_of_GPs
        self.delta_threshold = delta_threshold
        self.epsilon = epsilon
        
    def apply_smoothing(self, x):
        return uniform_filter(x, 7, mode="wrap")
    
    def derivative_gaussian(self, x):
        return gaussian_gradient_magnitude(x, sigma=0.5, mode='nearest') / 2
    
    def apply(self, I: np.ndarray):
        """
            Parameters:
            I: np.ndarray = RGB input image. Shape: (H x W x C)
            percentage_of_GPs: float = Percentage of gray pixels to select. 
            delta_threshold: float = Threshold for minimum difference in log differences.
        """
        h, w, c = I.shape
        num_pixels = h * w
        num_GPs = np.floor(self.percentage_of_GPs * num_pixels / 100).astype(int)
        
        
        R = I[:,:,0]; G = I[:,:,1]; B = I[:,:,2]
        M = (np.max(I, axis=-1) >= 0.95) | (np.sum(I, axis=-1) <= 0.0315)
        img_col = np.reshape(I, (num_pixels, c))

        R = self.apply_smoothing(R); G = self.apply_smoothing(G); B = self.apply_smoothing(B)
        M = M | (R == 0) | (G == 0) | (B == 0)
        R[R==0] = self.epsilon; G[G==0] = self.epsilon; B[B==0] = self.epsilon
        norm1 = R + G + B

        delta_R = self.derivative_gaussian(R)
        delta_G = self.derivative_gaussian(G)
        delta_B = self.derivative_gaussian(B)
        M = M | (delta_R <= self.delta_threshold) & (delta_G <= self.delta_threshold) & (delta_B <= self.delta_threshold)

        log_R = np.log(R) - np.log(norm1)
        log_B = np.log(B) - np.log(norm1)

        delta_log_R = self.derivative_gaussian(log_R)
        delta_log_B = self.derivative_gaussian(log_B)
        M = M | (delta_log_R == np.inf) | (delta_log_B == np.inf)
        
        delta = np.stack([
            np.reshape(delta_log_R, (h * w, -1)),
            np.reshape(delta_log_B, (h * w, -1))
        ], axis=-1)
        
        norm2 = np.linalg.norm(delta, axis=-1)
        uniq_lightmap = np.reshape(norm2, delta_log_R.shape)
        uniq_lightmap[M == 1] = np.max(uniq_lightmap)
        uniq_lightmap = self.apply_smoothing(uniq_lightmap)
        uniq_lightmap_flat = np.reshape(uniq_lightmap, (num_pixels))
        sorted_uniq_lightmap_flat = np.sort(uniq_lightmap_flat)

        unique_GIs = np.zeros_like(uniq_lightmap_flat)
        unique_GIs[uniq_lightmap_flat < sorted_uniq_lightmap_flat[num_GPs]] = 1
        mean_chosen_pixels = np.mean(img_col[unique_GIs == 1, :], axis=0)
        return mean_chosen_pixels / np.sqrt((mean_chosen_pixels**2).sum())
