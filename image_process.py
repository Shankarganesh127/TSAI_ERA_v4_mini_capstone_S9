class image_processing:
    
    def __init__(self):
        self.red = 0
        self.green = 1
        self.blue = 2
        self.mean_list = []

    def norm_image_to_red_green_blue_channels(self, image_array):
        img_norm = image_array.astype(np.float32) / 255.0
        npred = img_norm[:,:,self.red]
        npgreen = img_norm[:,:,self.green]
        npblue = img_norm[:,:,self.blue]
        return npred, npgreen, npblue

    def get_mean_list_of_surrounding_pixels(self, channel, i, j, kernel=1):
        loc_mean_list = []
        for di in [-kernel, 0, kernel]:
            for dj in [-kernel, 0, kernel]:
                ni, nj = i + di, j + dj
                if (0 <= ni < channel.shape[0]) and (0 <= nj < channel.shape[1]) and (di != 0 or dj != 0):
                    if channel[ni, nj] is not None:
                        loc_mean_list.append(channel[ni, nj])
        return loc_mean_list

    def get_pixel_value_mean_std(self, channel, kernel=1):
        channel_o = np.zeros_like(channel, dtype=float)
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                pixel_value = channel[i,j]
                mean_list = self.get_mean_list_of_surrounding_pixels(channel, i, j, kernel)
                surrounding_pixels_mean= np.mean(mean_list) if mean_list else 0
                surrounding_pixels_std = np.std(mean_list) if mean_list else 0
                #channel_o[i,j] = abs(pixel_value - surrounding_pixels_mean) #/ surrounding_pixels_std if surrounding_pixels_std else 0
                channel_o[i,j] = (pixel_value - surrounding_pixels_mean) / surrounding_pixels_std if surrounding_pixels_std else 0
                #channel_o[i,j] = (surrounding_pixels_mean - pixel_value) / surrounding_pixels_std if surrounding_pixels_std else 0
                #channel_o[i,j] = abs(surrounding_pixels_mean - pixel_value) #/ surrounding_pixels_std if surrounding_pixels_std else 0
        return channel_o

    def get_max_channel_index(self, rgb_image):
        max_channel_index = np.argmax(rgb_image, axis=2)
        return max_channel_index

    def get_processed_image(self, image_array, kernel=1):
        red, green, blue = self.norm_image_to_red_green_blue_channels(image_array)
        processed_channel_images_red = self.get_pixel_value_mean_std(red,kernel)
        processed_channel_images_green = self.get_pixel_value_mean_std(green,kernel)
        processed_channel_images_blue = self.get_pixel_value_mean_std(blue,kernel)
        processed_image = np.stack([processed_channel_images_red, processed_channel_images_green, processed_channel_images_blue], axis=-1)
        return processed_image

    def output_image_array(self, image_i, processed_channel_images):
        out_img_array = image_i[:,:,0] #np.zeros((processed_channel_images.shape[0], processed_channel_images.shape[1], 1), dtype=np.uint8)
        max_channel_index = self.get_max_channel_index(processed_channel_images)
        for i in range(image_i.shape[0]):
            for j in range(image_i.shape[1]):
                out_img_array[i,j] = image_i[i,j,max_channel_index[i,j]]
        return out_img_array
    
    def extract_image_features(self, numpy_image_array):
        processed_channel_images = self.get_processed_image(numpy_image_array) 
        final_image = self.output_image_array(numpy_image_array, processed_channel_images)
        return final_image