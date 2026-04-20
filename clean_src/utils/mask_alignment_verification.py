import rasterio

image_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-SPECTRAL_IMAGE.TIF"
mask_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CIRRUS.TIF"

with rasterio.open(image_path) as img, rasterio.open(mask_path) as msk:

    print("Shape image:", img.shape)
    print("Shape mask :", msk.shape)

    print("CRS image:", img.crs)
    print("CRS mask :", msk.crs)

    print("Transform image:", img.transform)
    print("Transform mask :", msk.transform)

    print("Resolution image:", img.res)
    print("Resolution mask :", msk.res)