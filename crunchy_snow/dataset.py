import torch
import torch.utils.data
import xarray as xr
import numpy as np

def calc_dowy(doy):
    'calculate day of water year from day of year'
    if doy < 274:
        dowy = doy + (365-274)
    elif doy >= 274:
        dowy = doy-274
    return dowy

def calc_norm(tensor, minmax_list):
    '''
    normalize a tensor between 0 and 1 using a min and max value stored in a list
    '''
    normalized = (tensor-minmax_list[0])/(minmax_list[1]-minmax_list[0])
    normalized = torch.nan_to_num(normalized, 0)
    return normalized

def undo_norm(tensor, minmax_list):
    '''
    undo tensor normalization
    '''
    original = (tensor*(minmax_list[1]-minmax_list[0]))+minmax_list[0]
    return original

def db_scale(x, epsilon=1e-10):
    # Add epsilon only where x is zero
    x_with_epsilon = np.where(x == 0, epsilon, x)
    # Calculate the logarithm
    log_x = 10 * np.log10(x_with_epsilon)
    # Set the areas where x was originally zero back to zero
    log_x[x == 0] = 0
    return log_x

# # find dataset min and max for normalization
# norm_dict = {}
# for i, outputs in enumerate(train_loader):
#     if (i+1)%1000 == 0: 
#         print(f'loop {i+1}/{train_data.filelength}')
#     for j, item in enumerate(outputs):
#         if i == 0:
#             norm_dict[j] = [item.min(), item.max()]
#         if item.max() > norm_dict[j][1]:
#             norm_dict[j][1] = item.max().item()
#         if item.min() < norm_dict[j][0] and not item.min() == 0:
#             norm_dict[j][0] = item.min().item()

# these are set by finding the min and max across the entire dataset
norm_dict = {'aso_sd':[0, 24.9],
             'vv':[0, 41.4],
             'vh':[0, 16.4],
             'cr':[],
             'delta_cr':[],
             'AOT':[0, 572.1],
             'coastal':[0, 23459.1],
             'blue':[0, 23004.1],
             'green':[0, 26440.1],
             'red':[0, 21576.1],
             'red_edge1':[0, 20796.1],
             'red_edge2':[0, 20432.1],
             'red_edge3':[0, 20149.1],
             'nir':[0, 21217.1],
             'water_vapor':[0, 18199.1],
             'swir1':[0, 17549.1],
             'swir2':[0, 17314.1],
             'scene_class_map':[0, 15],
             'water_vapor_product':[0, 6517.5],
             'elevation':[-100, 9000]}

# define dataset 
class Dataset(torch.utils.data.Dataset):
    '''
    class that reads data from a netCDF and returns normalized tensors 
    '''
    def __init__(self, path_list, selected_channels, norm_dict=norm_dict, norm=True):
        self.path_list = path_list
        self.selected_channels = selected_channels
        self.norm_dict = norm_dict
        self.norm = norm
        
    #dataset length
    def __len__(self):
        self.filelength = len(self.path_list)
        return self.filelength
    
    #load images
    def __getitem__(self,idx):
        ds = xr.open_dataset(self.path_list[idx])
        #ds = ds.coarsen(x = 6, boundary = 'trim').mean().coarsen(y = 6, boundary = 'trim').mean()
        # convert to tensors
        aso_sd = torch.from_numpy(np.float32(ds.aso_sd.values))
        snowon_vv = torch.from_numpy(db_scale(np.float32(ds.snowon_vv.values)))
        snowon_vh = torch.from_numpy(db_scale(np.float32(ds.snowon_vh.values)))
        snowoff_vv = torch.from_numpy(db_scale(np.float32(ds.snowoff_vv.values)))
        snowoff_vh = torch.from_numpy(db_scale(np.float32(ds.snowoff_vh.values)))
        snowon_vv_mean = torch.from_numpy(db_scale(np.float32(ds.snowon_vv_mean.values)))
        snowon_vh_mean = torch.from_numpy(db_scale(np.float32(ds.snowon_vh_mean.values)))
        snowoff_vv_mean = torch.from_numpy(db_scale(np.float32(ds.snowoff_vv_mean.values)))
        snowoff_vh_mean = torch.from_numpy(db_scale(np.float32(ds.snowoff_vh_mean.values)))
        aerosol_optical_thickness = torch.from_numpy(np.float32(ds.AOT.values))
        coastal_aerosol = torch.from_numpy(np.float32(ds.B01.values))
        blue = torch.from_numpy(np.float32(ds.B02.values))
        green = torch.from_numpy(np.float32(ds.B03.values))
        red = torch.from_numpy(np.float32(ds.B04.values))
        red_edge1 = torch.from_numpy(np.float32(ds.B05.values))
        red_edge2 = torch.from_numpy(np.float32(ds.B06.values))
        red_edge3 = torch.from_numpy(np.float32(ds.B07.values))
        nir = torch.from_numpy(np.float32(ds.B08.values))
        water_vapor = torch.from_numpy(np.float32(ds.B09.values))
        swir1 = torch.from_numpy(np.float32(ds.B11.values))
        swir2 = torch.from_numpy(np.float32(ds.B12.values))
        scene_class_map = torch.from_numpy(np.float32(ds.SCL.values))
        water_vapor_product = torch.from_numpy(np.float32(ds.WVP.values))
        fcf = torch.from_numpy(np.float32(ds.fcf.values))
        elevation = torch.from_numpy(np.float32(ds.elevation.values))
        aso_gap_map = torch.from_numpy(np.float32(ds.aso_gap_map.values))
        rtc_gap_map = torch.from_numpy(np.float32(ds.rtc_gap_map.values))
        rtc_mean_gap_map = torch.from_numpy(np.float32(ds.rtc_mean_gap_map.values))

        # calculate some other inputs for our CNN
        ndvi = (nir - red)/(nir + red)
        ndsi = (green - swir1)/(green + swir1)
        ndwi = (green - nir)/(green + nir)

        snowon_cr = snowon_vh - snowon_vv
        snowoff_cr = snowoff_vh - snowoff_vh
        delta_cr = snowon_cr - snowoff_cr

        # fn = os.path.split(path)[-1]
        # dowy_1d = calc_dowy(pd.to_datetime(fn.split('_')[4]).dayofyear)
        # dowy = torch.full_like(aso_sd, dowy_1d)
            
        # normalize layers (except gap maps and fcf)
        if self.norm == True:
            aso_sd = calc_norm(aso_sd, self.norm_dict['aso_sd'])
            snowon_vv = calc_norm(snowon_vv, self.norm_dict['vv'])
            snowon_vh = calc_norm(snowon_vh, self.norm_dict['vh'])
            snowoff_vv = calc_norm(snowoff_vv, self.norm_dict['vv'])
            snowoff_vh = calc_norm(snowoff_vh, self.norm_dict['vh'])
            snowon_vv_mean = calc_norm(snowon_vv_mean, self.norm_dict['vv'])
            snowon_vh_mean = calc_norm(snowon_vh_mean, self.norm_dict['vh'])
            snowoff_vv_mean = calc_norm(snowoff_vv_mean, self.norm_dict['vv'])
            snowoff_vh_mean = calc_norm(snowoff_vh_mean, self.norm_dict['vh'])
            aerosol_optical_thickness = calc_norm(aerosol_optical_thickness, self.norm_dict['AOT'])
            coastal_aerosol = calc_norm(coastal_aerosol, self.norm_dict['coastal'])
            blue = calc_norm(blue, self.norm_dict['blue'])
            green = calc_norm(green, self.norm_dict['green'])
            red = calc_norm(red, self.norm_dict['red'])
            red_edge1 = calc_norm(red_edge1, self.norm_dict['red_edge1'])
            red_edge2 = calc_norm(red_edge2, self.norm_dict['red_edge2'])
            red_edge3 = calc_norm(red_edge3, self.norm_dict['red_edge3'])
            nir = calc_norm(nir, self.norm_dict['nir'])
            water_vapor = calc_norm(water_vapor, self.norm_dict['water_vapor'])
            swir1 = calc_norm(swir1, self.norm_dict['swir1'])
            swir2 = calc_norm(swir2, self.norm_dict['swir2'])
            scene_class_map = calc_norm(scene_class_map, self.norm_dict['scene_class_map'])
            water_vapor_product = calc_norm(water_vapor_product, self.norm_dict['water_vapor_product'])
            elevation = calc_norm(elevation, self.norm_dict['elevation'])
            ndvi = torch.nan_to_num(calc_norm(ndvi, [-1, 1]), 0)
            ndsi = torch.nan_to_num(calc_norm(ndsi, [-1, 1]), 0)
            ndwi = torch.nan_to_num(calc_norm(ndwi, [-1, 1]), 0)
            snowon_cr = torch.nan_to_num(calc_norm(snowon_cr, self.norm_dict['cr']), 0)
            snowoff_cr = torch.nan_to_num(calc_norm(snowoff_cr, self.norm_dict['cr']), 0)
            delta_cr = torch.nan_to_num(calc_norm(delta_cr, self.norm_dict['delta_cr']), 0)

        data_dict = {'aso_sd':aso_sd[None, :, :],
                    'snowon_vv': snowon_vv[None, :, :],
                    'snowon_vh': snowon_vh[None, :, :],
                    'snowoff_vv': snowoff_vv[None, :, :],
                    'snowoff_vh': snowoff_vh[None, :, :],
                    'snowon_vv_mean': snowon_vv_mean[None, :, :],
                    'snowon_vh_mean': snowon_vh_mean[None, :, :],
                    'snowoff_vv_mean': snowoff_vv_mean[None, :, :],
                    'snowoff_vh_mean': snowoff_vh_mean[None, :, :],
                    'aerosol_optical_thickness': aerosol_optical_thickness[None, :, :],
                    'coastal_aerosol':coastal_aerosol[None, :, :],
                    'blue': blue[None, :, :],
                    'green': green[None, :, :],
                    'red': red[None, :, :],
                    'red_edge1': red_edge1[None, :, :],
                    'red_edge2': red_edge2[None, :, :],
                    'red_edge3': red_edge3[None, :, :],
                    'nir': nir[None, :, :],
                    'water_vapor': water_vapor[None, :, :],
                    'swir1': swir1[None, :, :],
                    'swir2': swir2[None, :, :],
                    'scene_class_map': scene_class_map[None, :, :],
                    'water_vapor_product': water_vapor_product[None, :, :],
                    'fcf': fcf[None, :, :],
                    'elevation': elevation[None, :, :],
                    'ndvi': ndvi[None, :, :],
                    'ndsi': ndsi[None, :, :],
                    'ndwi': ndwi[None, :, :],
                    'snowon_cr': snowon_cr[None, :, :],
                    'snowoff_cr': snowoff_cr[None, :, :],
                    'delta_cr': delta_cr[None, :, :],
                    'aso_gap_map': aso_gap_map[None, :, :],
                    'rtc_gap_map': rtc_gap_map[None, :, :],
                    'rtc_mean_gap_map': rtc_mean_gap_map[None, :, :]}

        # Select only the specified channels
        selected_data = [data_dict[channel] for channel in self.selected_channels]
        
        return tuple(selected_data)
