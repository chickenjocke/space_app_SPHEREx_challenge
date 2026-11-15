from astropy.wcs import WCS
import numpy as np

# Load data from image
image_data = hdulist["IMAGE"].data
spectral_wcs = WCS(header=hdulist["IMAGE"].header, fobj=hdulist, key="W")
spectral_wcs.sip = None 

# Generate pixel coordinate grids
(y, x) = np.indices(image_data.shape)

wavelength, bandwidth = spectral_wcs.pixel_to_world(x, y)
