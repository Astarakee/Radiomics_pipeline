import SimpleITK as itk



def read_nifti(image_path):
    """
    Parameters
    ----------
    image_path : Full directory to the image data.

    Returns
    -------
    img_itk : the itk image.
    img_size: the tensor size of the itk image.
    img_spacing : image spacing of the read itk image.
    img_origin : the origin coordinate of the read itk image

    """
    
    img_itk = itk.ReadImage(image_path)
    img_size = img_itk.GetSize()
    img_spacing = img_itk.GetSpacing() 
    img_origin = img_itk.GetOrigin()
    img_direction = img_itk.GetDirection()
    
    return img_itk, img_size, img_spacing, img_origin, img_direction


def get_subject_sequence(img_itk, img_size, img_spacing, img_origin, mask_direction):
    """
    Get the sequence from 4D image images
    Parameters
    ----------
    img_itk : ITK image
        ITK image of the original 4D image.
    img_size : tuple/list
        Representing the size of the original 4D image.
    img_spacing : tuple/list
        Representing the voxel resolution and slice thickness.
    img_origin : tuple/list
        origin coordinates off the image space.

    Returns
    -------
    itk_sequences : list
        Each item of the list is one sequence of the 4D image in
        ITK image format with the same spacing and origin of the main
        image.

    """     
    n_sequence = img_size[-1]
    img_array = itk.GetArrayFromImage(img_itk)
    itk_sequences = []
    for item in range(n_sequence):
        img_sequence = img_array[item]
        itk_img_sequence = itk.GetImageFromArray(img_sequence)
        itk_img_sequence.SetOrigin(img_origin[:3])
        itk_img_sequence.SetSpacing(img_spacing[:3])
        itk_img_sequence.SetDirection(mask_direction)
        itk_sequences.append(itk_img_sequence)
    
    return itk_sequences

