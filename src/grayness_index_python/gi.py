import numpy as np
import torch

from typing import Union
from torch.nn import functional as F
from scipy.ndimage import uniform_filter, gaussian_gradient_magnitude
from kornia.filters import box_blur


class GraynessIndex:
    """Estimates scene illuminant using the Grayness Index method.
    
    This implementation follows the original paper's methodology to identify
    gray pixels in an image and use them for illuminant estimation. Supports
    both NumPy (CPU) and PyTorch (GPU) implementations.
    
    Attributes:
        percentage_of_GPs (float): Percentage of pixels to consider as gray (0-100)
        delta_threshold (float): Minimum gradient magnitude threshold for gray pixels
        epsilon (float): Small value to prevent division by zero in log calculations
    """
    
    def __init__(
        self,
        percentage_of_GPs: float = 0.1,
        delta_threshold: float = 1e-4,
        epsilon: float = 1e-7
    ) -> None:
        """Initializes Grayness Index estimator with configuration parameters.
        
        Args:
            percentage_of_GPs: Percentage of pixels to select as gray candidates (0-100)
            delta_threshold: Minimum intensity gradient threshold for candidate selection
            epsilon: Stabilizing constant for logarithmic calculations
        """
        self.percentage_of_GPs = percentage_of_GPs
        self.delta_threshold = delta_threshold
        self.epsilon = epsilon
        
    def apply_smoothing_np(self, x: np.ndarray) -> np.ndarray:
        """Applies uniform filtering to 2D array (NumPy implementation).
        
        Uses circular padding to maintain spatial dimensions.
        
        Args:
            x: Input 2D array (H x W)
            
        Returns:
            Smoothed array with same dimensions as input
        """
        return uniform_filter(x, 7, mode="wrap")
    
    def derivative_gaussian_np(self, x: np.ndarray) -> np.ndarray:
        """Computes Gaussian gradient magnitude (NumPy implementation).
        
        Args:
            x: Input 2D array (H x W)
            
        Returns:
            Gradient magnitude array with same dimensions as input
        """
        return gaussian_gradient_magnitude(x, sigma=0.5, mode='nearest') / 2

    def apply_np(self, I: np.ndarray) -> np.ndarray:
        """Processes RGB image array to estimate illuminant (NumPy version).
        
        Args:
            I: Input RGB image (H x W x 3) in [0,255] or [0,1] range
            
        Returns:
            Estimated illuminant vector (3,) normalized to unit length
        """
        if I.dtype == np.uint8:
            I = I.astype(np.float32) / 255.0
        elif np.max(I) > 1.0:
            I = I.astype(np.float32) / 65535.0

        h, w, c = I.shape
        num_pixels = h * w
        num_GPs = np.floor(self.percentage_of_GPs * num_pixels / 100).astype(int)
        
        R = I[:,:,0]; G = I[:,:,1]; B = I[:,:,2]
        M = (np.max(I, axis=-1) >= 0.95) | (np.sum(I, axis=-1) <= 0.0315)
        img_col = np.reshape(I, (num_pixels, c))

        R = self.apply_smoothing_np(I[:,:,0])
        G = self.apply_smoothing_np(I[:,:,1]) 
        B = self.apply_smoothing_np(I[:,:,2])
        zero_mask = (R < 1e-3) | (G < 1e-3) | (B < 1e-3)
        M = M | zero_mask
        
        R = np.clip(R, self.epsilon, None)
        G = np.clip(G, self.epsilon, None)
        B = np.clip(B, self.epsilon, None)
        norm1 = R + G + B
        
        delta_R = self.derivative_gaussian_np(R)
        delta_G = self.derivative_gaussian_np(G)
        delta_B = self.derivative_gaussian_np(B)
        M = M | (delta_R <= self.delta_threshold) & (delta_G <= self.delta_threshold) & (delta_B <= self.delta_threshold)
        log_R = np.log(R) - np.log(norm1)
        log_B = np.log(B) - np.log(norm1)

        delta_log_R = self.derivative_gaussian_np(log_R)
        delta_log_B = self.derivative_gaussian_np(log_B)
        
        delta = np.stack([
            np.reshape(delta_log_R, (h * w, -1)),
            np.reshape(delta_log_B, (h * w, -1))
        ], axis=-1)
        norm2 = np.linalg.norm(delta, axis=-1)

        uniq_lightmap = np.reshape(norm2, delta_log_R.shape)
        uniq_lightmap[M == 1] = np.max(uniq_lightmap)
        uniq_lightmap = self.apply_smoothing_np(uniq_lightmap)
        uniq_lightmap_flat = np.reshape(uniq_lightmap, (num_pixels))

        sorted_uniq_lightmap_flat = np.sort(uniq_lightmap_flat)
        unique_GIs = np.zeros_like(uniq_lightmap_flat)
        img_col = np.reshape(I, (num_pixels, c))
        if num_GPs > 0:
            threshold_value = sorted_uniq_lightmap_flat[num_GPs-1]
            unique_GIs[uniq_lightmap_flat <= threshold_value] = 1
        chosen_pixels = img_col[unique_GIs == 1, :]
        mean_chosen_pixels = np.mean(chosen_pixels, axis=0)

        return mean_chosen_pixels / np.sqrt((mean_chosen_pixels**2).sum())
    
    def apply_smoothing_T(self, x: torch.Tensor) -> torch.Tensor:
        """Applies box blur filtering to 4D tensor (PyTorch implementation).
        
        Uses circular padding and matches SciPy's uniform filter behavior.
        
        Args:
            x: Input tensor (B x C x H x W)
            
        Returns:
            Smoothed tensor with same dimensions as input
        """
        return box_blur(x, kernel_size=(7, 7), border_type='circular')
    
    def derivative_gaussian_T(self, x: torch.Tensor) -> torch.Tensor:
        """Computes Gaussian gradient magnitude (PyTorch implementation).
        
        Matches SciPy's gaussian_gradient_magnitude behavior exactly.
        
        Args:
            x: Input tensor (B x C x H x W)
            
        Returns:
            Gradient magnitude tensor (B x C x H x W)
        """
        sigma = 0.5
        kernel_size = 5
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        
        x_coord = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size-1) // 2
        gaussian = torch.exp(-x_coord**2 / (2 * sigma**2))
        gaussian_deriv = (-x_coord / sigma**2) * gaussian
        
        gaussian = gaussian / gaussian.sum()
        gaussian_deriv = gaussian_deriv / gaussian_deriv.abs().sum()

        kernel_x = gaussian_deriv.view(1, 1, -1, 1) * gaussian.view(1, 1, 1, -1)
        kernel_y = gaussian.view(1, 1, -1, 1) * gaussian_deriv.view(1, 1, 1, -1)
        
        gradient_x = F.conv2d(
            x, 
            kernel_x.repeat(x.shape[1], 1, 1, 1),
            padding='same',
            groups=x.shape[1],
            bias=None
        )
        gradient_y = F.conv2d(
            x, 
            kernel_y.repeat(x.shape[1], 1, 1, 1),
            padding='same',
            groups=x.shape[1],
            bias=None
        )
        gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)

        return gradient_magnitude / 2.0
    
    def apply_T(self, I: torch.Tensor) -> torch.Tensor:
        """Processes batch of RGB tensors to estimate illuminant (PyTorch version).
        
        Args:
            I: Input tensor (B x C x H x W) or (C x H x W) in [0,255] or [0,1] range
            
        Returns:
            Estimated illuminant vectors (B x 3) normalized to unit length
        """
        if I.dtype == torch.uint8:
            I = I.float() / 255.0
        elif I.max() > 1.0:
            I = I.float() / 65535.0

        batch_size, c, h, w = I.shape
        num_pixels = h * w
        num_GPs = int(self.percentage_of_GPs * num_pixels / 100)
        
        R = I[:,0,:,:].unsqueeze(1)
        G = I[:,1,:,:].unsqueeze(1)
        B = I[:,2,:,:].unsqueeze(1)
        M = (I.amax(dim=1, keepdim=True) >= 0.95) | (I.sum(dim=1, keepdim=True) <= 0.0315)
        img_col = I.view(batch_size, c, -1)

        R = self.apply_smoothing_T(R)
        G = self.apply_smoothing_T(G)
        B = self.apply_smoothing_T(B)
        zero_mask = (R < 1e-3) | (G < 1e-3) | (B < 1e-3)
        M = M | zero_mask

        R = torch.clamp(R, min=self.epsilon)
        G = torch.clamp(G, min=self.epsilon)
        B = torch.clamp(B, min=self.epsilon)
        norm1 = (R + G + B)
        
        delta_R = self.derivative_gaussian_T(R)
        delta_G = self.derivative_gaussian_T(G)
        delta_B = self.derivative_gaussian_T(B)
        M = M | (delta_R <= self.delta_threshold) & (delta_G <= self.delta_threshold) & (delta_B <= self.delta_threshold)
        log_R = torch.log(R) - torch.log(norm1)
        log_B = torch.log(B) - torch.log(norm1)
        
        delta_log_R = self.derivative_gaussian_T(log_R)
        delta_log_B = self.derivative_gaussian_T(log_B)
        M = M | delta_log_R.isinf() | delta_log_B.isinf()
        
        delta = torch.stack([
            delta_log_R.view(batch_size, -1),
            delta_log_B.view(batch_size, -1)
        ], dim=-1)
        
        norm2 = torch.linalg.norm(delta, dim=-1)
        
        uniq_lightmap = norm2.view(batch_size, h, w)
        max_val = torch.amax(uniq_lightmap, dim=(1,2), keepdim=True)
        mask = M.squeeze(1)
        uniq_lightmap = torch.where(mask, max_val, uniq_lightmap)
        uniq_lightmap = uniq_lightmap.unsqueeze(1)
        uniq_lightmap = self.apply_smoothing_T(uniq_lightmap)
        uniq_lightmap = uniq_lightmap.squeeze(1)
        
        uniq_lightmap_flat = uniq_lightmap.squeeze(1).view(batch_size, -1)
        sorted_vals = torch.sort(uniq_lightmap_flat, dim=-1).values
        threshold = sorted_vals[:, num_GPs-1].unsqueeze(-1)
        unique_GIs = (uniq_lightmap_flat <= threshold).float()
        
        chosen_pixels = img_col * unique_GIs.unsqueeze(1).bool()
        mean_chosen_pixels = torch.mean(chosen_pixels, dim=-1)
        return mean_chosen_pixels / torch.sqrt((mean_chosen_pixels**2).sum(dim=-1, keepdim=True))

    def apply(self, I: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Unified interface for illuminant estimation from various input types.
        
        Automatically routes to numpy or pytorch implementation based on input type.
        
        Args:
            I: Input image (H x W x 3) or batch (B x 3 x H x W)
            
        Returns:
            Estimated illuminant(s) matching input type and dimensions
        """
        if isinstance(I, np.ndarray):
            assert I.ndim == 3, "Number of dimensions of Numpy image should be 3."
            assert I.shape[-1] == 3, "Number of color channels of Numpy image should be 3."
            return self.apply_np(I)
        elif isinstance(I, torch.Tensor):
            if I.ndim == 3:
                I = I.unsqueeze(0)
            if I.shape[-1] == 3:
                I = I.permute(0, 3, 1, 2)
            if torch.cuda.is_available():
                I = I.cuda()
            return self.apply_T(I)
        else:
            raise NotImplementedError("Image should be represented as Numpy array or Torch tensor.")
