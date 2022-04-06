"""Functions to query the DB for FINALCUT lightcurves."""

from datetime import date

from astropy.coordinates import SkyCoord
import easyaccess as ea
import numpy as np
import pandas as pd


def get_lc(ra: float, dec: float, rad: float = 2.0) -> pd.DataFrame:
    """Query Y6A1_FINALCUT_OBJECT at location and construct lightcurve.
    
    Args:
      ra (float): The RA value to query.
      dec (float): The DEC value to query.
      rad (float, default=2.0): The box search size in arcsec.

    Returns:
      A DataFrame containing the lightcurve.
    """

    # Connect to DB.
    conn = ea.connect(section='dessci', quiet=True)
    cursor = conn.cursor()

    # Build up query.
    columns = [
        'o.FLUX_APER_12', 'o.FLUXERR_APER_12', 'o.FLUX_AUTO', 'o.FLUXERR_AUTO',
        'o.FLUX_RADIUS', 'o.NITE', 'o.EXPNUM', 'o.RA', 'o.DEC', 'o.BAND', 
        'o.BACKGROUND', 'o.IMAFLAGS_ISO', 'z.MAG_ZERO', 'z.SIGMA_MAG_ZERO',
    ]
    cols = ', '.join(columns)
    size = rad / 3600.0  # Convert to deg.
    cond = f"o.RA between {ra - size} and {ra + size} and o.DEC between {dec - size} and {dec + size} and o.IMAFLAGS_ISO = 0 and o.filename = z.catalogname"
    query = f"SELECT {cols} FROM Y6A1_FINALCUT_OBJECT o, Y6A1_ZEROPOINT z WHERE {cond} ORDER BY o.NITE, o.BAND"

    # Query DB.
    _ = cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    df = pd.DataFrame(data=rows, columns=columns)

    # Format lightcurve.
    nites = df['o.NITE'].values.astype(str)
    df['DATE'] = [date.fromisoformat(f"{n[0:4]}-{n[4:6]}-{n[6:]}") for n in nites]
    db_coords = SkyCoord(ra=df['o.RA'].values, dec=df['o.DEC'].values, unit='deg')
    coadd_coord = SkyCoord(ra=ra, dec=dec, unit='deg')
    df['SEPARATION'] = coadd_coord.separation(db_coords).arcsec

    df['BS_FLUX_APER_12'] = df['o.FLUX_APER_12'].values - np.pi * 144 * df['o.BACKGROUND'].values
    df['BS_FLUX_AUTO'] = df['o.FLUX_AUTO'].values - np.pi * df['o.FLUX_RADIUS'].values**2 * df['o.BACKGROUND'].values
    df['FLUX_APER_12'] = np.where(df['o.FLUX_APER_12'].values < 1.0, 1.0, df['o.FLUX_APER_12'].values)
    df['FLUX_AUTO'] = np.where(df['o.FLUX_AUTO'].values < 1.0, 1.0, df['o.FLUX_AUTO'].values)
    df['BS_FLUX_APER_12'] = np.where(df['BS_FLUX_APER_12'].values < 1.0, 1.0, df['BS_FLUX_APER_12'].values)
    df['BS_FLUX_AUTO'] = np.where(df['BS_FLUX_AUTO'].values < 1.0, 1.0, df['BS_FLUX_AUTO'].values)
    
    df['MAG_AUTO'] = df['z.MAG_ZERO'].values - 2.5 * np.log10(df['FLUX_AUTO'].values)
    df['BS_MAG_AUTO'] = df['z.MAG_ZERO'].values - 2.5 * np.log10(df['BS_FLUX_AUTO'].values)
    df['MAG_APER_12'] = df['z.MAG_ZERO'].values - 2.5 * np.log10(df['FLUX_APER_12'].values)
    df['BS_MAG_APER_12'] = df['z.MAG_ZERO'].values - 2.5 * np.log10(df['BS_FLUX_APER_12'].values)

    df['MAGERR_AUTO'] = np.sqrt((2.5 * np.log10(df['FLUX_AUTO'].values / (df['FLUX_AUTO'].values + np.abs(df['o.FLUXERR_AUTO'].values))))**2 + df['z.SIGMA_MAG_ZERO'].values**2)
    df['BS_MAGERR_AUTO'] = np.sqrt((2.5 * np.log10(df['BS_FLUX_AUTO'].values / (df['BS_FLUX_AUTO'].values + np.abs(df['o.FLUXERR_AUTO'].values))))**2 + df['z.SIGMA_MAG_ZERO'].values**2)
    df['MAGERR_APER_12'] = np.sqrt((2.5 * np.log10(df['FLUX_APER_12'].values / (df['FLUX_APER_12'].values + np.abs(df['o.FLUXERR_APER_12'].values))))**2 + df['z.SIGMA_MAG_ZERO'].values**2)
    df['BS_MAGERR_APER_12'] = np.sqrt((2.5 * np.log10(df['BS_FLUX_APER_12'].values / (df['BS_FLUX_APER_12'].values + np.abs(df['o.FLUXERR_APER_12'].values))))**2 + df['z.SIGMA_MAG_ZERO'].values**2)

    return df


# Testing.
if __name__ == "__main__":
    ra, dec = 36.589451, -3.896260
    df = get_lc(ra, dec)

    print(df.columns)
    print(df.shape)
    print(df.head())

    print(type(df['DATE'].values[0]))
    print(type(df['SEPARATION'].values[0]))