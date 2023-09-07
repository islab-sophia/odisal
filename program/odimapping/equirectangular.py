from .base_mapping import BaseMapping
import numpy as np

def equirectangular(mapping_type, maps_dir, embed_overlapping='average',
                    view_angle_p=100, view_angle_t=100,
                    extract_h=500, extract_w=500,
                    odi_h=800, odi_w=1600, odi_c=3):
    """
    Parameters
    -----------
    mapping_type : str
        name of mapping method
    maps_dir : str
        directory of map file
    embed_overlapping : str
        default : average
        overlapping method
    view_angle_p : float
        default : 100
        view angle phi of extracted image
    view_angle_t : float
        default : 100
        view angle theta of extracted image
    extract_h : int
        default : 500
        height of extracted image
    extract_w : int
        default : 500
        width of extracted image
    odi_h : int
        default : 800
        height of equirectangular image
    odi_w : int
        default : 1600
        width of equirectangular image
    odi_c : int
        default : 3 (RGB)
        num of channels of equirectangular image

    Returns
    -----------
    mapping : object
    """

    if mapping_type=='E6':
        mapping = E6(embed_overlapping, maps_dir, view_angle_p, view_angle_t, extract_h, extract_w, odi_h, odi_w, odi_c)
    if mapping_type=='E14':
        mapping = E14(embed_overlapping, maps_dir, view_angle_p, view_angle_t, extract_h, extract_w, odi_h, odi_w, odi_c)
    if mapping_type=='E26':
        mapping = E26(embed_overlapping, maps_dir, view_angle_p, view_angle_t, extract_h, extract_w, odi_h, odi_w, odi_c)
    if mapping_type=='E62':
        mapping = E62(embed_overlapping, maps_dir, view_angle_p, view_angle_t, extract_h, extract_w, odi_h, odi_w, odi_c)
    if mapping_type=='E114':
        mapping = E114(embed_overlapping, maps_dir, view_angle_p, view_angle_t, extract_h, extract_w, odi_h, odi_w, odi_c)
    return mapping

def _equirectangular_E(
        name, p_list, t_list,
        embed_overlapping, maps_dir, view_angle_p=100, view_angle_t=100, extract_h=500, extract_w=500, odi_h=800, odi_w=1600, odi_c=3):
    """
    Parameters
    -----------
    name : str
    p_list : list of float or int
    t_list : list of float or int
    embed_overlapping : str
    maps_dir : str
    view_angle_p : float
    view_angle_t : float
    extract_h : int
    extract_w : int
    odi_h : int
    odi_w : int
    odi_c : int
    """

    view_angle = (np.radians(view_angle_p), np.radians(view_angle_t))

    num_extracted = len(p_list)*len(t_list)+2
    num_vertical=len(p_list)+2
    num_horizontal=len(t_list)

    camera_list=np.zeros((num_extracted, 2))
    camera_list[0] = [90, 0]
    camera_list[1] = [-90, 0]
    i=2
    for p in p_list:
        for t in t_list:
            camera_list[i] = [p, t]
            i += 1
    camera_list = np.array(camera_list)

    mapping = BaseMapping(
                camera_list=camera_list,
                input_method='equirectangular',
                name=name,
                embed_overlapping=embed_overlapping,
                extract_size=(extract_h, extract_w),
                odi_size=(odi_h, odi_w),
                odi_c=odi_c,
                view_angle=view_angle,
                num_vertical=num_vertical,
                num_extracted=num_extracted,
                num_horizontal=num_horizontal,
                maps_dir=maps_dir)

    return mapping

def E6(embed_overlapping, maps_dir, view_angle_p, view_angle_t, extract_h, extract_w, odi_h, odi_w, odi_c):
    """
    extract 6 images from ODI
    90 degree interval
    this is cube mapping

    [setting]
    p_list : angles of vertical direction
    t_list : angles of horizontal direction
    name : name of this extracting method

    Parameters
    -----------
    embed_overlapping : str
    maps_dir : str
    view_angle_p : float
    view_angle_t : float
    extract_h : int
    extract_w : int
    odi_h : int
    odi_w : int
    odi_c : int

    Returns
    -----------
    mapping : object
    """
    mapping = _equirectangular_E(
                name='E6',
                p_list=[0],
                t_list=[180, 90, 0, -90],
                embed_overlapping=embed_overlapping,
                maps_dir=maps_dir,
                view_angle_p=view_angle_p,
                view_angle_t=view_angle_t,
                extract_h=extract_h,
                extract_w=extract_w,
                odi_h=odi_h,
                odi_w=odi_w,
                odi_c=odi_c)
    return mapping

def E14(embed_overlapping, maps_dir, view_angle_p, view_angle_t, extract_h, extract_w, odi_h, odi_w, odi_c):
    """
    extract 14 images from ODI
    60 degree interval

    [setting]
    p_list : angles of vertical direction
    t_list : angles of horizontal direction
    name : name of this extracting method

    Parameters
    -----------
    embed_overlapping : str
    maps_dir : str
    view_angle_p : float
    view_angle_t : float
    extract_h : int
    extract_w : int
    odi_h : int
    odi_w : int
    odi_c : int

    Returns
    -----------
    mapping : object
    """
    mapping = _equirectangular_E(
                name='E14',
                p_list=[30, -30],
                t_list=[180, 120, 60, 0, -60, -120],
                embed_overlapping=embed_overlapping,
                maps_dir=maps_dir,
                view_angle_p=view_angle_p,
                view_angle_t=view_angle_t,
                extract_h=extract_h,
                extract_w=extract_w,
                odi_h=odi_h,
                odi_w=odi_w,
                odi_c=odi_c)
    return mapping

def E26(embed_overlapping, maps_dir, view_angle_p, view_angle_t, extract_h, extract_w, odi_h, odi_w, odi_c):
    """
    extract 26 images from ODI
    45 degree interval
    this is recomended method in paper

    [setting]
    p_list : angles of vertical direction
    t_list : angles of horizontal direction
    name : name of this extracting method

    Parameters
    -----------
    embed_overlapping : str
    maps_dir : str
    view_angle_p : float
    view_angle_t : float
    extract_h : int
    extract_w : int
    odi_h : int
    odi_w : int
    odi_c : int

    Returns
    -----------
    mapping : object
    """
    mapping = _equirectangular_E(
                name='E26',
                p_list=[45, 0, -45],
                t_list=[180, 135, 90, 45, 0, -45, -90, -135],
                embed_overlapping=embed_overlapping,
                maps_dir=maps_dir,
                view_angle_p=view_angle_p,
                view_angle_t=view_angle_t,
                extract_h=extract_h,
                extract_w=extract_w,
                odi_h=odi_h,
                odi_w=odi_w,
                odi_c=odi_c)
    return mapping

def E62(embed_overlapping, maps_dir, view_angle_p, view_angle_t, extract_h, extract_w, odi_h, odi_w, odi_c):
    """
    extract 62 images from ODI
    30 degree interval

    [setting]
    p_list : angles of vertical direction
    t_list : angles of horizontal direction
    name : name of this extracting method

    Parameters
    -----------
    embed_overlapping : str
    maps_dir : str
    view_angle_p : float
    view_angle_t : float
    extract_h : int
    extract_w : int
    odi_h : int
    odi_w : int
    odi_c : int

    Returns
    -----------
    mapping : object
    """
    mapping = _equirectangular_E(
                name='E62',
                p_list=[60, 30, 0, -30, -60],
                t_list=[180, 150, 120, 90, 60, 30, 0, -30, -60, -90, -120, -150],
                embed_overlapping=embed_overlapping,
                maps_dir=maps_dir,
                view_angle_p=view_angle_p,
                view_angle_t=view_angle_t,
                extract_h=extract_h,
                extract_w=extract_w,
                odi_h=odi_h,
                odi_w=odi_w,
                odi_c=odi_c)
    return mapping

def E114(embed_overlapping, maps_dir, view_angle_p, view_angle_t, extract_h, extract_w, odi_h, odi_w, odi_c):
    """
    extract 114 images from ODI
    22.5 degree interval

    [setting]
    p_list : angles of vertical direction
    t_list : angles of horizontal direction
    name : name of this extracting method

    Parameters
    -----------
    embed_overlapping : str
    maps_dir : str
    view_angle_p : float
    view_angle_t : float
    extract_h : int
    extract_w : int
    odi_h : int
    odi_w : int
    odi_c : int

    Returns
    -----------
    mapping : object
    """
    mapping = _equirectangular_E(
                name='E114',
                p_list=[67.5, 45, 22.5, 0, -22.5, -45, -67.5],
                t_list=[180, 157.5, 135, 112.5, 90, 67.5, 45, 22.5, 0, -22.5, -45, -67.5, -90, -112.5, -135, -157.5],
                embed_overlapping=embed_overlapping,
                maps_dir=maps_dir,
                view_angle_p=view_angle_p,
                view_angle_t=view_angle_t,
                extract_h=extract_h,
                extract_w=extract_w,
                odi_h=odi_h,
                odi_w=odi_w,
                odi_c=odi_c)
    return mapping
