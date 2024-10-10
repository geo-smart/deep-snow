import torch
import torch.utils.data
import xarray as xr
import numpy as np
import os
import pandas as pd
import torchvision.transforms.functional 
import random
from deep_snow.utils import calc_dowy, calc_norm, undo_norm, db_scale

# these are set by finding the min and max across the entire dataset
norm_dict = {'aso_sd':[0, 25],
             'vv':[-59, 30],
             'vh':[-65, 17],
             'cr':[-43, 16],
             'delta_cr':[-33, 27],
             'AOT':[0, 572],
             'coastal':[0, 24304],
             'blue':[0, 23371],
             'green':[0, 26440],
             'red':[0, 21576],
             'red_edge1':[0, 20796],
             'red_edge2':[0, 20432],
             'red_edge3':[0, 20149],
             'nir':[0, 21217],
             'water_vapor':[0, 18199],
             'swir1':[0, 17669],
             'swir2':[0, 17936],
             'scene_class_map':[0, 15],
             'water_vapor_product':[0, 6518],
             'elevation':[-100, 9000],
             'aspect':[0, 360],
             'slope':[0, 90],
             'curvature':[-22, 22],
             'tpi':[-164, 167],
             'tri':[0, 913],
             'latitude':[-90, 90],
             'longitude':[-180, 180]}

def random_transform(image, randoms):
    # Random horizontal flip
    if randoms[0] > 0.5:
        image = torchvision.transforms.functional.hflip(image)
    # Random vertical flip
    if randoms[1] > 0.5:
        image = torchvision.transforms.functional.vflip(image)
    # Random rotation
    angles = [0, 90, 180, 270]
    angle = angles[randoms[2]]
    image = torchvision.transforms.functional.rotate(image, angle)
    return image

# define dataset 
class Dataset(torch.utils.data.Dataset):
    '''
    class that reads data from a netCDF and returns normalized tensors 
    '''
    def __init__(self, path_list, selected_channels, norm_dict=norm_dict, norm=True, augment=True):
        self.path_list = path_list
        self.selected_channels = selected_channels
        self.norm_dict = norm_dict
        self.norm = norm
        self.augment = augment
        
    #dataset length
    def __len__(self):
        self.filelength = len(self.path_list)
        return self.filelength
    
    #load images
    def __getitem__(self,idx):
        ds = xr.open_dataset(self.path_list[idx])
        # to downsample dataset
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
        slope = torch.from_numpy(np.float32(ds.slope.values))
        aspect = torch.from_numpy(np.float32(ds.aspect.values))
        curvature = torch.from_numpy(np.float32(ds.curvature.values))
        tri = torch.from_numpy(np.float32(ds.tri.values))
        tpi = torch.from_numpy(np.float32(ds.tpi.values))
        latitude = torch.from_numpy(np.float32(ds.latitude.values))
        longitude = torch.from_numpy(np.float32(ds.longitude.values))
        aso_gap_map = torch.from_numpy(np.float32(ds.aso_gap_map.values))
        rtc_gap_map = torch.from_numpy(np.float32(ds.rtc_gap_map.values))
        rtc_mean_gap_map = torch.from_numpy(np.float32(ds.rtc_mean_gap_map.values))
        s2_gap_map = torch.from_numpy(np.float32(ds.s2_gap_map.values))

        # calculate some other inputs for our CNN
        ndvi = torch.nan_to_num((nir - red)/(nir + red), 0)
        ndsi = torch.nan_to_num((green - swir1)/(green + swir1), 0)
        ndwi = torch.nan_to_num((green - nir)/(green + nir), 0)

        snowon_cr = snowon_vh - snowon_vv
        snowoff_cr = snowoff_vh - snowoff_vv
        delta_cr = snowon_cr - snowoff_cr

        fn = os.path.split(self.path_list[idx])[-1]
        dowy_1d = calc_dowy(pd.to_datetime(fn.split('_')[4]).dayofyear)
        dowy = torch.full_like(aso_sd, dowy_1d)
            
        # normalize layers (except gap maps and fcf)
        if self.norm == True:
            aso_sd = torch.clamp(calc_norm(aso_sd, self.norm_dict['aso_sd']), 0, 1)
            snowon_vv = torch.clamp(calc_norm(snowon_vv, self.norm_dict['vv']), 0, 1)
            snowon_vh = torch.clamp(calc_norm(snowon_vh, self.norm_dict['vh']), 0, 1)
            snowoff_vv = torch.clamp(calc_norm(snowoff_vv, self.norm_dict['vv']), 0, 1)
            snowoff_vh = torch.clamp(calc_norm(snowoff_vh, self.norm_dict['vh']), 0, 1)
            snowon_vv_mean = torch.clamp(calc_norm(snowon_vv_mean, self.norm_dict['vv']), 0, 1)
            snowon_vh_mean = torch.clamp(calc_norm(snowon_vh_mean, self.norm_dict['vh']), 0, 1)
            snowoff_vv_mean = torch.clamp(calc_norm(snowoff_vv_mean, self.norm_dict['vv']), 0, 1)
            snowoff_vh_mean = torch.clamp(calc_norm(snowoff_vh_mean, self.norm_dict['vh']), 0, 1)
            aerosol_optical_thickness = torch.clamp(calc_norm(aerosol_optical_thickness, self.norm_dict['AOT']), 0, 1)
            coastal_aerosol = torch.clamp(calc_norm(coastal_aerosol, self.norm_dict['coastal']), 0, 1)
            blue = torch.clamp(calc_norm(blue, self.norm_dict['blue']), 0, 1)
            green = torch.clamp(calc_norm(green, self.norm_dict['green']), 0, 1)
            red = torch.clamp(calc_norm(red, self.norm_dict['red']), 0, 1)
            red_edge1 = torch.clamp(calc_norm(red_edge1, self.norm_dict['red_edge1']), 0, 1)
            red_edge2 = torch.clamp(calc_norm(red_edge2, self.norm_dict['red_edge2']), 0, 1)
            red_edge3 = torch.clamp(calc_norm(red_edge3, self.norm_dict['red_edge3']), 0, 1)
            nir = torch.clamp(calc_norm(nir, self.norm_dict['nir']), 0, 1)
            water_vapor = torch.clamp(calc_norm(water_vapor, self.norm_dict['water_vapor']), 0, 1)
            swir1 = torch.clamp(calc_norm(swir1, self.norm_dict['swir1']), 0, 1)
            swir2 = torch.clamp(calc_norm(swir2, self.norm_dict['swir2']), 0, 1)
            scene_class_map = torch.clamp(calc_norm(scene_class_map, self.norm_dict['scene_class_map']), 0, 1)
            water_vapor_product = torch.clamp(calc_norm(water_vapor_product, self.norm_dict['water_vapor_product']), 0, 1)
            elevation = torch.clamp(calc_norm(elevation, self.norm_dict['elevation']), 0, 1)
            aspect = torch.clamp(calc_norm(aspect, self.norm_dict['aspect']), 0, 1)
            slope = torch.clamp(calc_norm(slope, self.norm_dict['slope']), 0, 1)
            curvature = torch.clamp(calc_norm(curvature, self.norm_dict['curvature']), 0, 1)
            tpi = torch.clamp(calc_norm(tpi, self.norm_dict['tpi']), 0, 1)
            tri = torch.clamp(calc_norm(tri, self.norm_dict['tri']), 0, 1)
            latitude = torch.clamp(calc_norm(latitude, self.norm_dict['latitude']), 0, 1)
            longitude = torch.clamp(calc_norm(longitude, self.norm_dict['longitude']), 0, 1)
            dowy = torch.clamp(torch.nan_to_num(calc_norm(dowy, [0, 365]), 0), 0, 1)
            ndvi = torch.clamp(torch.nan_to_num(calc_norm(ndvi, [-1, 1]), 0), 0, 1)
            ndsi = torch.clamp(torch.nan_to_num(calc_norm(ndsi, [-1, 1]), 0), 0, 1)
            ndwi = torch.clamp(torch.nan_to_num(calc_norm(ndwi, [-1, 1]), 0), 0, 1)
            snowon_cr = torch.clamp(torch.nan_to_num(calc_norm(snowon_cr, self.norm_dict['cr']), 0), 0, 1)
            snowoff_cr = torch.clamp(torch.nan_to_num(calc_norm(snowoff_cr, self.norm_dict['cr']), 0), 0, 1)
            delta_cr = torch.clamp(torch.nan_to_num(calc_norm(delta_cr, self.norm_dict['delta_cr']), 0), 0, 1)

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
                    'slope': slope[None, :, :],
                    'aspect': aspect[None, :, :],
                    'curvature': curvature[None, :, :],
                    'tpi': tpi[None, :, :],
                    'tri': tri[None, :, :],
                    'latitude': latitude[None, :, :],
                    'longitude': longitude[None, :, :],
                    'dowy': dowy[None, :, :],
                    'ndvi': ndvi[None, :, :],
                    'ndsi': ndsi[None, :, :],
                    'ndwi': ndwi[None, :, :],
                    'snowon_cr': snowon_cr[None, :, :],
                    'snowoff_cr': snowoff_cr[None, :, :],
                    'delta_cr': delta_cr[None, :, :],
                    'aso_gap_map': aso_gap_map[None, :, :],
                    'rtc_gap_map': rtc_gap_map[None, :, :],
                    'rtc_mean_gap_map': rtc_mean_gap_map[None, :, :],
                    's2_gap_map': s2_gap_map[None, :, :]}

        # Select only the specified channels
        selected_data = [data_dict[channel] for channel in self.selected_channels]

        # Apply transformations to each selected channel
        if self.augment:
            randoms = [random.random(), random.random(), random.randint(0, 3)]
            selected_data = [random_transform(img, randoms) for img in selected_data]
        
        return tuple(selected_data)

