from shapely.geometry import Polygon
import random
from tqdm import tqdm
import geopandas
import argparse
import os

from owslib.wms import WebMapService
from constants import WMS, MAP_NAME

import uuid

ZONES = [
    "Rock",
    "Improved grassland",
    "Wetland",
    "Peatland",
    "Heather",
    "Water",
    "Grassland",
    "Urban",
    "Woodland",
    "Arable and horticulture",
]


def get_higher_bound(coord, pixels, res=0.25):
    return pixels * res + coord


def generate_backgrounds(data, zone, args, pbar):

    # Generate output path
    out_path = os.path.join(args.output_path, "_".join(zone.split(" ")))
    os.makedirs(out_path, exist_ok=True)

    # Get zone data
    polygons = data[data["broad_LC"] == zone]["geometry"]
    min_x, min_y, max_x, max_y = polygons.total_bounds

    # Sample patches
    print(f"...{zone}")

    i = 0
    while i < args.n_samples:
        # Generate candidate patch
        w, s = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
        e = get_higher_bound(w, args.patch_px, args.res)
        n = get_higher_bound(s, args.patch_px, args.res)
        gen_poly = Polygon(((w, s), (w, n), (e, n), (e, s)))

        # Check patch
        if polygons.contains(gen_poly).any():
            # Save patch
            try:
                img = get_map_from_ws(gen_poly.bounds, polygons.crs.to_epsg())
                out = open(
                    os.path.join(out_path, f"S{uuid.uuid4()}.jpg"),
                    "wb",
                )
                out.write(img.read())
                out.close()

                pbar.update()
                i += 1
            except:
                pass


def convert_epsg_coord(coord, epsg1=4326, epsg2=27700):
    w, s, e, n = coord
    gdf = geopandas.GeoDataFrame(
        {
            "col1": ["name1"],
            "geometry": [Polygon(((w, s), (w, n), (e, n), (e, s)))],
        },
        crs=f"EPSG:{epsg1}",
    )
    gdf = gdf.to_crs(f"EPSG:{epsg2}")
    return gdf.total_bounds


def get_map_from_ws(coordinates, epsg="3857", res=0.25):
    """
    bbox -> west, south, east, north
    Height (in pixels) = (north-south)/0.25
    Width (in pixels) = (east-west)/0.25
    """
    w, s, e, n = coordinates

    height = int(((n - s) / res))
    width = int(((e - w) / res))

    img = wms.getmap(
        layers=[MAP_NAME],
        srs=f"EPSG:{epsg}",
        bbox=(w, s, e, n),
        size=(width, height),
        format="image/jpeg",
        transparent=False,
    )

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-zone_ids",
        help='Zones to include from sampling area. "Rock":1, "Improved grassland":2, "Wetland":3, "Peatland":4, "Heather":5, "Water":6, "Grassland":7, "Urban":8, "Woodland":9, "Arable and horticulture":10',
        type=str,
        default="1,2,3,4,5,6,7,8,9,10",
    )

    parser.add_argument(
        "-n_samples",
        help="Number of backgrund samples to generate",
        type=int,
        default=100,
    )

    parser.add_argument(
        "-area_bounds",
        help="Coordinates for specifying the area to sample from",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-patch_px",
        help="Patch size in pixels",
        type=int,
        default=224,
    )

    parser.add_argument(
        "-res",
        help="Aerial image resolution in m",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "-save_coord",
        help="Save the coordinates of the images as .txt file",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "-data_path",
        help="Path to the landcover shapefile.",
        type=str,
        default="../WMS/GB_landcover_vector_250m/GB_landcover_simplified_polygons_250m.shp",
    )

    parser.add_argument(
        "-output_path",
        help="Path to the the saving folder",
        type=str,
        default="../Datasets/SampledBackgrounds",
    )

    # Variables initialisation
    args = parser.parse_args()
    wms = WebMapService(WMS)

    ids = [int(id) for id in args.zone_ids.split(",")]
    zones_ids = [ZONES[i - 1] for i in ids]

    custom_bounds = None
    if args.area_bounds:
        custom_bounds = [float(coord) for coord in args.area_bounds.split(",")]

        min_x, min_y, max_x, max_y = convert_epsg_coord(custom_bounds, 4326, 27700)

    print("Loading shapefile...")
    data = geopandas.read_file(args.data_path)

    print("Sampling patches...")
    pbar = tqdm(total=args.n_samples * len(zones_ids))
    for zone in zones_ids:

        # Generate output path
        out_path = os.path.join(args.output_path, "_".join(zone.split(" ")))
        os.makedirs(out_path, exist_ok=True)

        # Get zone data
        polygons = data[data["broad_LC"] == zone]["geometry"]

        if custom_bounds is None:
            min_x, min_y, max_x, max_y = polygons.total_bounds

        # Sample patches
        print(f"...{zone}")

        i = 0
        while i < args.n_samples:
            # Generate candidate patch
            w, s = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
            e = get_higher_bound(w, args.patch_px, args.res)
            n = get_higher_bound(s, args.patch_px, args.res)
            gen_poly = Polygon(((w, s), (w, n), (e, n), (e, s)))

            # Check patch
            if polygons.contains(gen_poly).any():
                # Save patch
                try:
                    # Save image
                    img = get_map_from_ws(gen_poly.bounds, polygons.crs.to_epsg())
                    img_id = uuid.uuid4()
                    out = open(
                        os.path.join(out_path, f"S{img_id}.jpg"),
                        "wb",
                    )
                    out.write(img.read())
                    out.close()

                    # Save coordinates
                    f = open(os.path.join(out_path, f"S{img_id}.txt"), "w")
                    w2, s2, e2, n2 = convert_epsg_coord(
                        [w, s, e, n], epsg1=27700, epsg2=4326
                    )
                    f.write(f"{w2},{s2},{e2},{n2}")
                    f.close()

                    pbar.update()
                    i += 1
                except:
                    pass
