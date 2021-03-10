import torch
import numpy as np
from . import utils


__all__ = ["two_dwt_on_3d"]


def two_dwt_on_3d(image, decomp_levl=3, qmf=None, fwt="FWT_PO_1D_2D_3D_fast"):
    """
    # % usage
    % [y, s1, s2]=twoDWTon3Ddata(x,L,qmf,FWT)
    % Take the 2D-DWT for each band and reshape subbands as [wLL(:);wLH(:);wHL(:);wHH(:)];
    % input
    % x: 3D data
    % L: level of dicomposition
    % qmf: quadrature mirror filter
    % FWT: either 'mdwt', or 'FWT_PO_1D_2D_3D'. mdwt is defualt which is a mex
    % and faster if there is any difficulty choose FWT_PO_1D_2D_3D
    %
    % output
    % y: vactorized 2D ortho-wavelet coefficient for each band as [wLL(:);wLH(:);wHL(:);wHH(:)];
    % s1: the number of rows
    % s2: the number of columns

    Parameters
    ----------
    image
    decomp_levl
    qmf
    fwt

    Returns
    -------

    """
    s1, s2, s3 = image.shape
    if qmf is None:
        qmf = utils.daubcqf(4)

    ret = torch.zeros(s1 * s2, s3)

    if fwt == "wavedec2":
        raise NotImplementedError("wavedec2 not implemented")
        # todo: implement the wavedec2 option, not needed yet, see matlab code below
        # if strcmpi(FWT,'wavedec2')
        #     s1=[];
        #     for i=1:s3
        #         [y1,S_S] = wavedec2(image(:,:,i),L,['db',num2str(length(qmf)+2)]);
        #         y(:,i)=vec(y1);
        #         s1(:,:,i)=S_S;
        #     end

    for i in range(s3):
        ydum2 = None
        nx, ny = s1, s2
        
    # for i=1:s3
    #     ydum2=[];
    #     nx=s1;
    #     ny=s2;
    #     y1=eval([FWT '(image(:,:,i),qmf,L)']);
    #     ydum=y1;
    #     for j=1:L
    #         y2 = mat2cell(ydum,[nx/2,nx/2],[ny/2,ny/2]);
    #         wLH=cell2mat(y2(1,2));
    #         wHL=cell2mat(y2(2,1));
    #         wHH=cell2mat(y2(2,2));
    #         ydum2= [wLH(:);wHL(:);wHH(:);ydum2];
    #         nx=nx/2;ny=ny/2;
    #         ydum=cell2mat(y2(1,1));
    #     end
    #     y(:,i)=[ydum(:);ydum2];
    # end

