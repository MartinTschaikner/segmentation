"""

This class returns a 4D stack of ring scan(s) or of interpolated ring scan(s) from the following file types:
npy, vol, img
in case of volume file types (vol, img) the bmo text files for BMO points are needed as well.
"""
from os import listdir
import sys
from tkinter.filedialog import askdirectory
import OCT_read
import ring_scan_from_vol
import oct_img_reader
import ring_scan_from_img
import numpy as np


class Return4DRingScanStack:
    """

    This class will return a 4d ring scan stack of (an) original or interpolated ring scan(s)
    """

    def __init__(self, input_file):

        self.input_file = input_file

    def return_4d_stack(self, radius, number_circle_points, filter_parameter):

        file = self.input_file.split('/')[-3:]
        file_name = file[0] + '/' + file[1] + '/' + file[2]
        file_extension = file[2].split('.')[-1]

        if file_extension == 'npy':
            if file_name.find('Optic Disc Cube') != -1:
                print('Cirrus ring scan(s). No BMO data needed. Segmentation starts!')
                ring_scan_stack = np.load(file_name)
                if len(ring_scan_stack.shape) == 2:
                    ring_scan_stack = np.expand_dims(ring_scan_stack, axis=0)
                if len(ring_scan_stack.shape) == 3:
                    ring_scan_stack = np.expand_dims(ring_scan_stack, axis=3)

            else:
                print('Spectralis ring scan(s). No BMO data needed. Segmentation starts!')
                ring_scan_stack = np.load(file_name)
                if len(ring_scan_stack.shape) == 2:
                    ring_scan_stack = np.expand_dims(ring_scan_stack, axis=0)
                if len(ring_scan_stack.shape) == 3:
                    ring_scan_stack = np.expand_dims(ring_scan_stack, axis=3)

        elif file_extension == 'vol':
            if file[2].split('_')[-1] == '0.vol':
                print('Spectralis ring scan. No BMO data needed!')
                # Get the all information about the input ring scan file
                oct_info_ring_scan = OCT_read.OctRead(file_name)
                # Get the header of the input ring scan file
                header_ring_scan = oct_info_ring_scan.get_oct_hdr()
                # Get the b scan stack of the input ring scan file
                b_scan_stack = oct_info_ring_scan.get_b_scans(header_ring_scan)
                # Get needed data
                ring_scan = b_scan_stack.reshape(header_ring_scan['SizeZ'], header_ring_scan['SizeX'])
                ring_scan_stack = np.expand_dims(ring_scan, axis=0)
                ring_scan_stack = np.expand_dims(ring_scan_stack, axis=3)

            else:
                print('Interpolating Spectralis volume file to ring scan!')
                bmo = askdirectory(title='Choose folder with Spectralis bmo data!')
                bmo = bmo.split('/')[-2:]
                bmo_path = bmo[0] + '/' + bmo[1] + '/'
                bmo_file_name = [f for f in listdir(bmo_path) if f.find(file[2].split('.')[0]) != -1][0]
                # Get the all information about the input ring scan file
                oct_info_vol = OCT_read.OctRead(file_name)
                # Get the header of the input ring scan file
                header_vol = oct_info_vol.get_oct_hdr()
                # Get the b scan stack of the input ring scan file
                b_scan_stack = oct_info_vol.get_b_scans(header_vol)
                # compute interpolated grey values, ilm and rpe segmentation
                ring_scan_interpolated = \
                    ring_scan_from_vol.RingScanFromVolume(header_vol, b_scan_stack, bmo_path + bmo_file_name,
                                                          radius, number_circle_points, filter_parameter)
                # compute correct circle points to corresponding scan pattern (OS vs OD)
                circle_points = ring_scan_interpolated.circle_points_coordinates()
                # compute interpolated grey values, ilm and rpe segmentation
                ring_scan_int = \
                    ring_scan_interpolated.ring_scan_interpolation(circle_points)
                # output 4D array
                ring_scan_stack = np.expand_dims(ring_scan_int, axis=0)
                ring_scan_stack = np.expand_dims(ring_scan_stack, axis=3)

        elif file_extension == 'img':
            print('Interpolating Cirrus img volume file to ring scan!')
            bmo = askdirectory(title='Choose folder with Cirrus bmo data!')
            bmo = bmo.split('/')[-2:]
            bmo_path = bmo[0] + '/' + bmo[1] + '/'
            bmo_file_name = [f for f in listdir(bmo_path) if f.find(file[2].split('_')[-3]) != -1][0]
            # hardcoded for Cirrus img file
            cube_size = np.array([6, 6, 2])
            # oct object
            oct_cirrus = oct_img_reader.OctReadImg(file_name, cube_size)
            # object info
            cell_info, cell_entries = oct_cirrus.get_cell_info()
            vol_info, vol_size, cube = oct_cirrus.get_vol_info_and_size(cell_info, cell_entries)
            num_a_scans, num_b_scans = oct_cirrus.get_volume_dimension(cube, vol_info, vol_size)
            volume_data, volume_resolution = oct_cirrus.get_image_data(num_a_scans, num_b_scans)
            volume_resolution = np.roll(volume_resolution, 2)
            ring_scan_interpolated = ring_scan_from_img.\
                RingScanFromImg(volume_data, bmo_path + bmo_file_name, volume_resolution, cell_info[4], radius,
                                number_circle_points, filter_parameter)

            circle_points = ring_scan_interpolated.circle_points_coordinates()

            ring_scan_int = ring_scan_interpolated.ring_scan_interpolation(circle_points)
            ring_scan_stack = np.expand_dims(ring_scan_int, axis=0)
            ring_scan_stack = np.expand_dims(ring_scan_stack, axis=3)

        else:
            print('Data type unknown!')
            sys.exit()

        return ring_scan_stack
