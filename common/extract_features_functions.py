'''
This Module simply contains functions to extract features from pre-trained models
or do white box feature extraction

'''

from Image_File_IO.extract_features_iterator import extract_features_iterator


def extract_features_white_box(DIRECTORY_PATH,
                               includedCategories=['Dress', 'Skirt', 'UpperBody', 'LowerBody'],
                               imageReshape = None,
                               extractor_functions = None):

    extract_features_iterator(DIRECTORY_PATH,
                              includedCategories = includedCategories,
                              imageReshape = imageReshape,
                              extractor_functions=extractor_functions,
                              isWhiteboxExtraction=True)



def extract_features_pre_trained(DIRECTORY_PATH,
                                 model,
                                 layer_name = None,
                                 layer_index = None,
                                 includedCategories=['Dress', 'Skirt', 'UpperBody', 'LowerBody'],
                                 imageReshape = 224):
    extract_features_iterator(DIRECTORY_PATH,
                              model=model,
                              layer_name=layer_name,
                              layer_index = layer_index,
                              includedCategories = includedCategories,
                              imageReshape = imageReshape,
                              isWhiteboxExtraction=False)





