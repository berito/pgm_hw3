import os
from sentinelhub import SentinelHubSession,SentinelHubProductRequest, SentinelHubCatalogClient, MimeType, CRS, BBox

def download_sentinel2_image(output_dir, date, bbox, sh_client_id, sh_client_secret, resolution=10):
   
    # Set up Sentinel-Hub session
    session = SentinelHubSession(
        is_debug=False,
        sh_client_id=sh_client_id,
        sh_client_secret=sh_client_secret
    )

    # Create the Sentinel-Hub input parameters
    input_data = SentinelHubProductRequest(
        bbox=BBox(bbox, crs=CRS.WGS84),
        time_interval=(date, date),
        resolution=resolution,
        maxcc=0.2  # Maximum cloud cover percentage
    )

    # Retrieve the Sentinel-2 image
    catalog_client = SentinelHubCatalogClient(session)
    image_info = catalog_client.get_dataset(input_data, MimeType.jpeg)

    # Save the image to disk
    output_path = os.path.join(output_dir, f'sentinel2_{date}.tif')
    with open(output_path, 'wb') as file:
        file.write(image_info.content)

    print(f'Sentinel-2 image saved to: {output_path}')


output_dir = './'
sh_client_id="fc5c56d7-a7a6-4831-bf0e-60ebd4de80e3"
sh_client_secret="fFVCvGBtK9JjSE93nA8SLdXaC7Z55w2B"
date = '2024-05-21'
bbox = (14.2, 40.5, 14.4, 40.7)  # Bounding box in WGS84 coordinates

download_sentinel2_image(output_dir, date, bbox)