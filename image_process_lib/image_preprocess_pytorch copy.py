import torch
import torch.nn.functional as F

class image_processing:
    def __init__(self):
        self.red = 0
        self.green = 1
        self.blue = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def norm_image_to_red_green_blue_channels(self, image_tensor):
        """Normalize image tensor to [0, 1] and split into RGB channels."""
        # Ensure input is float and normalized to [0, 1]
        img_norm = image_tensor.float() / 255.0
        # Split into RGB channels (assuming shape [B, C, H, W] or [C, H, W])
        red = img_norm[:, self.red:self.red+1, :, :]  # Keep channel dim
        green = img_norm[:, self.green:self.green+1, :, :]
        blue = img_norm[:, self.blue:self.blue+1, :, :]
        return red, green, blue

    def get_pixel_value_mean_std(self, channel, kernel=1):
        """Compute mean and std of surrounding pixels using convolution."""
        # Ensure channel is [B, 1, H, W]
        if channel.dim() == 3:
            channel = channel.unsqueeze(0)  # Add batch dim if needed

        # Create a uniform kernel for computing local mean (3x3 or larger based on kernel)
        kernel_size = 2 * kernel + 1
        kernel_weights = torch.ones(1, 1, kernel_size, kernel_size, device=channel.device)
        # Set center of kernel to 0 to exclude the pixel itself
        kernel_weights = kernel_weights.clone()  # Clone to avoid in-place operation issues
        kernel_weights[:, :, kernel, kernel] = 0
        # Normalize kernel (excluding center pixel)
        kernel_weights = kernel_weights / kernel_weights.sum()

        # Compute local mean using convolution
        mean = F.conv2d(channel, kernel_weights, padding=kernel)

        # Compute local std: std = sqrt(E[X^2] - (E[X])^2)
        squared = channel ** 2
        mean_squared = F.conv2d(squared, kernel_weights, padding=kernel)
        variance = mean_squared - (mean ** 2)
        # Clamp variance to avoid negative values due to numerical errors
        variance = torch.clamp(variance, min=1e-8)
        std = torch.sqrt(variance)

        # Compute output: (pixel - mean) / std
        output = (channel - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
        return output

    def get_max_channel_index(self, rgb_image):
        """Get index of the channel with maximum value."""
        # rgb_image: [B, 3, H, W]
        max_channel_index = torch.argmax(rgb_image, dim=1, keepdim=True)  # [B, 1, H, W]
        return max_channel_index

    def get_processed_image(self, image_tensor, kernel=1):
        """Process image to compute normalized channels."""
        # Move tensor to device
        image_tensor = image_tensor.to(self.device)
        
        # Normalize and split into RGB channels
        red, green, blue = self.norm_image_to_red_green_blue_channels(image_tensor)
        
        # Process each channel
        processed_red = self.get_pixel_value_mean_std(red, kernel)
        processed_green = self.get_pixel_value_mean_std(green, kernel)
        processed_blue = self.get_pixel_value_mean_std(blue, kernel)
        
        # Stack processed channels: [B, 3, H, W]
        processed_image = torch.cat([processed_red, processed_green, processed_blue], dim=1)
        return processed_image

    def output_image_array(self, image_tensor, processed_image):
        """Extract pixel values based on max channel index."""
        # Move inputs to device
        image_tensor = image_tensor.to(self.device)
        processed_image = processed_image.to(self.device)
        
        # Get max channel index: [B, 1, H, W]
        max_channel_index = self.get_max_channel_index(processed_image)
        
        # Gather pixel values from original image based on max channel index
        # image_tensor: [B, 3, H, W], max_channel_index: [B, 1, H, W]
        output = torch.gather(image_tensor, dim=1, index=max_channel_index)  # [B, 1, H, W]
        return output

    def extract_image_features(self, input_tensor, kernel=1):
        """Main method to process batched or single images."""
        # Handle both batched ([B, C, H, W]) and single ([C, H, W]) inputs
        input_was_3d = input_tensor.dim() == 3
        if input_was_3d:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dim: [1, C, H, W]
        
        # Process image
        processed_image = self.get_processed_image(input_tensor,kernel)
        final_image = self.output_image_array(input_tensor, processed_image)
        
        # Remove batch dim if input was single image
        if input_was_3d:
            final_image = final_image.squeeze(0)  # [1, H, W] -> [H, W]
        
        return final_image