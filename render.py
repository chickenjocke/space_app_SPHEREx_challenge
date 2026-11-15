import http.client
import time
import urllib.error
from astropy.io import fits

max_retries = 3
spectral_image_url = "YOUR_TARGET_SPHEREx_FITS_URL"

for attempt in range(max_retries):
    try:
        with fits.open(spectral_image_url, memmap=False) as hdulist:
            for hdu in hdulist:
                hdu.data
        break
    except (TimeoutError, urllib.error.HTTPError, http.client.IncompleteRead):
        if attempt == max_retries - 1:
            raise
        time.sleep(10 * (attempt + 1))
