import numpy as np

def overscan_subtraction(image,overscan,trim,combine_method,fit_deg):
    '''
    given an image with an overscan region, performs an overscan subtraction with a row-wise polynomial fit of the overscan region

    params:
        image - 2D array with image (including overscan region)
        overscan - list with [min_column,max_column] of overscan region (indexed from 0)
        trim - list with [min_column,max_column] of image region (indexed from 0)
        combine_method - method for collapsing columns of overscan region, "median" or "mean"
        fit_deg - degree of polynomial row-wise fit of overscan region
    returns:
        image_sub_trim - trimmed and overscan subtracted image
    '''

    # collect overscan portion of the image
    image_overscan=image[:,overscan[0]:overscan[1]]

    # collapse the columns of the overscan using median or mean
    if combine_method=='mean':
        combine_overscan=np.mean(image_overscan,axis=1)
    elif combine_method=='median':
        combine_overscan=np.median(image_overscan,axis=1)
    else:
        raise ValueError('combine_method must be mean or median')

    rows=np.arange(len(combine_overscan))


    # perform a polynomial fit across the rows of the combined overscan
    p=np.polyfit(x=rows,y=combine_overscan,deg=fit_deg).flip() #flipped so 0 order term is first
    beta=p.reshape((len(p),1))

    # Now create design matrix
    X=np.zeros((len(rows),fit_deg+1))
    for column in range(X.shape[1]):
        X[:,column]=rows**column

    # do some matrix multiplication to get the fit
    fit=np.matmul(X,beta).flatten() # flattened so it's not a column matrix

    image_subtract=np.zeros((image.shape[0],image.shape[1]))
    # now subtract out fit row-wise
    for row in range(len(rows)):
        image_subtract[row,:]=image[row,:]-fit[row]

    # finally trim off overscan
    image_sub_trim=image_subtract[:,trim[0]:trim[1]]

    return image_sub_trim
