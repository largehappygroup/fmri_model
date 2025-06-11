function nnw = write_nii(nii_template,img_mat,fname)
% function nn = write_nii_nih(nii_template,img_to_write,fname)
% write nii (3d or 4d) to nifti format, based on header info from
% nii_template (in same space)
%
% inputs:
% ---------
% * nii_template: a matlab nifti struct (e.g., as read in by
% load_untouch_nii); later extend to read form file
% * img_mat: matlab matrix or the image (3d or 4d ok)
% * fname: filename to save image

% checking image dims to ensure we have proper nii template
%assert(all(size(img_mat==size(nii_template.img)))
ndims = length(size(img_mat));
if ndims>4
    error('does not support >4 dimensions');
end

nnw = nii_template;
nnw.img = img_mat;
nnw.hdr.dime.datatype = 16;
nnw.hdr.dime.dim(1) = ndims; 
nnw.hdr.dime.dim(5) = size(img_mat,4);

% if 4D, make sure it's recognized as such 
if (ndims==4 && nnw.hdr.dime.pixdim(5)==0)
    nnw.hdr.dime.pixdim(5) = 1;
    display(['note: time-step in nifti was zero..arbitrarily setting to 1 for ' ...
             'this 4D image']);
end

% save image
save_untouch_nii(nnw,fname);


