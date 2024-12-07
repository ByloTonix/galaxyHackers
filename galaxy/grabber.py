"""Script to fetch cutouts from the legacy survey"""

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import wget

from galaxy import util
from galaxy.config import settings


def download_url(url: str, outfile: str, max_attempts: int = 100) -> bool:
    """Downloads a file from a given URL with retries in case of errors.

    Args:
        url (str): URL of the file to download.
        outfile (str): Output file path.
        max_attempts (int, optional): Maximum number of download attempts. Defaults to 100.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    # Often encounter the following error:
    # urllib.error.HTTPError: HTTP Error 504: Gateway Time-out
    # Repeat the download attempt for up to `max_attempts` tries
    # Return True if the download was successful
    for attempt in range(max_attempts):
        try:
            wget.download(url=url, out=outfile)
            return True
        except HTTPError as e:
            print(f"Failed attempt {attempt} to download {outfile} with an HTTPError")
        except URLError as e:
            print(f"Failed attempt {attempt} to download {outfile} with a URLError")
        time.sleep(1)

    print(f"Failed to download image {outfile}")
    return False


class Grabber:
    """Class for fetching image cutouts from the legacy survey."""

    def __init__(
        self,
        survey_layer: str | None = None,
        bands: str | None = None,
        imgsize_arcmin: float = 1.5,
        imgsize_pix: int = 224,
        extra_processing: Optional[Callable] = None,
        extra_processing_kwargs: Dict[Any, Any] = dict(),
    ):
        """
        Args:
            survey_layer (str, optional): Survey layer to use. Defaults to settings.LEGACY_SURVEY.LAYER.
            bands (str, optional): Bands to include in the cutouts. Defaults to settings.LEGACY_SURVEY.BANDS.
            imgsize_arcmin (float, optional): Size of the image in arcminutes. Defaults to 1.5.
            imgsize_pix (int, optional): Size of the image in pixels. Defaults to 224.
            extra_processing (Callable, optional): Additional processing function for fetched cutouts. Defaults to None.
            extra_processing_kwargs (Dict[Any, Any], optional): Additional arguments for the processing function. Defaults to None.
        """

        self.survey_layer = survey_layer or settings.LEGACY_SURVEY.LAYER
        self.bands = bands or settings.LEGACY_SURVEY.BANDS
        self.imgsize_arcmin = imgsize_arcmin
        self.imgsize_pix = imgsize_pix
        self.extra_processing = extra_processing
        self.extra_processing_kwargs = extra_processing_kwargs

    def make_url(
        self, ra: float, dec: float, s_arcmin: float = 3.0, s_px: int = 512, format: str = "fits"
    ) -> str:
        """Creates a URL for fetching an image cutout from the legacy survey.

        Args:
            ra (float): Right ascension in degrees.
            dec (float): Declination in degrees.
            s_arcmin (float, optional): Size of the image in arcminutes. Defaults to 3.0.
            s_px (int, optional): Size of the image in pixels. Defaults to 512.
            format (str, optional): Format of the output file. Defaults to "fits".

        Returns:
            str: URL for the cutout.
        """

        # Convert coords to string
        ra, dec = str(np.round(ra, 5)), str(np.round(dec, 5))

        # Set pixscale
        s_arcsec = 60 * s_arcmin
        pxscale = 0.262  # s_arcsec / s_px

        # Convert image scales to string
        s_px, pxscale = str(s_px), str(np.round(pxscale, 4))

        url = (
            f"http://legacysurvey.org/viewer/cutout.{format}"
            f"?ra={ra}&dec={dec}"
            f"&layer={self.survey_layer}&pixscale={pxscale}&size={s_px}"
        )

        if self.bands:
            url += f"&bands={self.bands}"

        return url

    def grab_cutout(self, ra: float, dec: float, output_path: Path) -> None:
        """Fetches a single image cutout.

        Args:
            ra (float): Right ascension in degrees.
            dec (float): Declination in degrees.
            output_path (Path): Output file path.
        """
        url = self.make_url(ra=ra, dec=dec)

        if not os.path.exists(output_path):
            status = download_url(url, output_path)
            if status and (self.extra_processing is not None):
                self.extra_processing(output_path, **self.extra_processing_kwargs)

    def grab_cutouts(
        self,
        targets: pd.DataFrame,
        name_col: str = "name",
        ra_col: str = "ra_deg",
        dec_col: str = "dec_deg",
        output_dir: str = "",
        suffix: str = "",
        file_format: str = "fits",
    ) -> None:
        """Function to download image cutouts from any survey.
        ​
            Arguments:
                target_file {str, pd.DataFrame} -- Input file or DataFrame containing the list of target
                                                coordinates and names.
        ​
            Keyword Arguments:
                name_col {str} -- The column name in target_file that contains the desired file name
                                (default: {"Component_name"})
                ra_col {str} -- RA column name (default: {"RA"})
                dec_col {str} -- Dec column name (default: {"DEC"})
                survey {str} -- Survey name to pass to the legacy server (default: {"vlass1.2"})
                output_dir {str} -- Output path for the image cutouts (default: {""})
                prefix {str} -- Prefix for the output filename (default {""})
                suffix {str} -- Suffix for the output filename (default {survey})
                imgsize_arcmin {float} -- Image angular size in arcminutes (default: {3.0})
                imgsize_pix {int} -- Image size in pixels (default: {500})
        """

        suffix = suffix or self.survey_layer

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        holds = []
        for idx, target in targets.iterrows():

            name = idx
            # name = target[name_col]
            a = target[ra_col]
            d = target[dec_col]

            outfile = os.path.join(output_dir, f"{name}.{file_format}")

            holds.append((a, d, outfile))

        bighold = list(util.divide_chunks(holds, 20))
        print(len(bighold))

        # TODO Seems suboptimal, maybe asyncio?
        for i in range(len(bighold)):
            jobs = []
            for j in range(0, len(bighold[i])):

                ra, dec, outfile = bighold[i][j]
                thread = threading.Thread(
                    target=self.grab_cutout,
                    args=(ra, dec, outfile),
                )
                jobs.append(thread)
                thread.start()
            for k in jobs:
                k.join()


# def load_catalogue(catalog, pandas=False):
#     fmt = "fits" if catalog.endswith("fits") else "csv"
#     rcat = Table.read(catalog, format=fmt)

#     if pandas:
#         rcat = rcat.to_pandas()
#         if fmt == "fits":
#             for col in rcat.columns[rcat.dtypes == object]:
#                 rcat[col] = rcat[col].str.decode("ascii")

#     return rcat


# def cadc_cutout_url(ql_url, coords, radius):
#     # Extract cutout url from the QL tiles hosts on CADC
#     standard_front = (
#         "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/sync?ID=ad%3AVLASS%2F"
#     )
#     # coords assumes astropy sky coord, radius now an astropy angle
#     # ql_url is the url of the CADC hosted 1sq deg QL image
#     encoded_ql = urllib.parse.quote(ql_url.split("/")[-1])
#     encoded_ql = encoded_ql.replace("%3F", "&").replace("?", "&")
#     cutout_end = (
#         f"&CIRCLE={coords.ra.value}+{coords.dec.value}+{radius.to(u.deg).value}"
#     )
#     return standard_front + encoded_ql + cutout_end


# def make_filename(name, format="fits"):
#     filename = f"{name}.{format}"
#     return filename


# def process_unwise(fname, band="w1"):
#     # Remove axis from unWISE files that contains the band info
#     banddict = {"w1": 0, "w2": 1}

#     hdu = fits.open(fname)
#     imdata = hdu[0].data[banddict[band]]

#     # Fix header
#     nhead = hdu[0].header.copy()
#     nhead["NAXIS"] = 2
#     nhead["BAND"] = band

#     delkeys = ["NAXIS3", "BANDS", "BAND0", "BAND1"]
#     for key in delkeys:
#         del nhead[key]

#     newhdu = fits.PrimaryHDU(imdata, header=nhead)
#     nhl = fits.HDUList(newhdu)
#     nhl.writeto(fname, overwrite=True)


# def process_vlass_image(infile, outfile, ext=0, scale_unit=True, sfactor=1000):
#     # Process the image to hdulist len==1 and 2D WCS header
#     # Not needed if obtained from legacy survey
#     hdu = fits.open(infile)
#     data = hdu[ext].data.squeeze()
#     header = hdu[ext].header

#     if scale_unit:
#         data = sfactor * data

#     # Fix header to 2D
#     hkeys = list(header.keys())

#     crkeys = ["CTYPE", "CRVAL", "CDELT", "CRPIX", "CUNIT"]
#     cr3 = [f"{c}3" for c in crkeys]
#     cr4 = [f"{c}4" for c in crkeys]
#     badkeys = cr3 + cr4 + ["NAXIS3", "NAXIS4"]

#     for key in hkeys:
#         if "PC3" in key or "PC4" in key or "_3" in key or "_4" in key:
#             badkeys.append(key)
#         if key in badkeys:
#             del header[key]

#     header["NAXIS"] = 2

#     # Write fits file
#     newhdu = fits.PrimaryHDU(data)
#     newhdu.header = header
#     nhlist = fits.HDUList(newhdu)
#     nhlist.writeto(outfile)


# def grab_vlass_cutouts(
#     target_file, output_dir=None, vlass_dir="", unwise_dir="", **kwargs,
# ):
#     if output_dir is not None:
#         vlass_dir = output_dir
#         unwise_dir = output_dir

#     grab_cutouts(
#         target_file, output_dir=vlass_dir, survey="vlass1.2", suffix="VLASS", **kwargs
#     )


# def grab_vlass_unwise_cutouts(
#     target_file, output_dir=None, vlass_dir="", unwise_dir="", **kwargs,
# ):
#     if output_dir is not None:
#         vlass_dir = output_dir
#         unwise_dir = output_dir

#     # Download VLASS cutouts
#     grab_cutouts(
#         target_file, output_dir=vlass_dir, survey="vlass1.2", suffix="VLASS", **kwargs
#     )

#     # Download unWISE cutouts
#     grab_cutouts(
#         target_file,
#         output_dir=unwise_dir,
#         survey="unwise-neo7",
#         suffix="unWISE-NEO7",
#         extra_processing=process_unwise,
#         extra_proc_kwds={"band": "w1"},
#         **kwargs,
#     )


# def parse_args():
#     """
#     Parse input arguments
#     """
#     parser = argparse.ArgumentParser(description="Download VLASS and unWISE cutouts.")
#     parser.add_argument("target_file", help="Path to the desired SOM to annotate")
#     parser.add_argument(
#         "-p",
#         "--path",
#         dest="path",
#         help="Directory for output files",
#         default=None,
#         type=str,
#     )
#     parser.add_argument(
#         "--vlass_path",
#         dest="vlass_dir",
#         help="Directory for output VLASS files",
#         default="",
#         type=str,
#     )
#     parser.add_argument(
#         "--unwise_path",
#         dest="unwise_dir",
#         help="Directory for output unWISE files",
#         default="",
#         type=str,
#     )
#     parser.add_argument(
#         "--img_size",
#         dest="img_size",
#         help="Image size in pixels",
#         default=300,
#         type=int,
#     )
#     parser.add_argument(
#         "--ang_size",
#         dest="ang_size",
#         help="Image angular size in arcminutes",
#         default=3.0,
#         type=float,
#     )
#     parser.add_argument(
#         "--name_col",
#         dest="name_col",
#         help="Name of the column containing the file name",
#         default="Component_name",
#         type=str,
#     )
#     parser.add_argument(
#         "--ra_col",
#         dest="ra",
#         help="Name of the RA column in the input catalogue",
#         default="RA",
#         type=str,
#     )
#     parser.add_argument(
#         "--dec_col",
#         dest="dec",
#         help="Name of the Dec column in the input catalogue",
#         default="DEC",
#         type=str,
#     )

#     args = parser.parse_args()
#     return args


# if __name__ == "__main__":

#     args = parse_args()

#     if args.path is not None:
#         args.vlass_dir = args.path
#         args.unwise_dir = args.path

#     grab_vlass_cutouts(
#         args.target_file,
#         name_col=args.name_col,
#         ra_col=args.ra,
#         dec_col=args.dec,
#         vlass_dir=args.vlass_dir,
#         unwise_dir=args.unwise_dir,
#         imgsize_arcmin=args.ang_size,
#         imgsize_pix=args.img_size,
#     )
