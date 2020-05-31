%[L,Centers] = imsegkmeans(I1,2);
%B = labeloverlay(I1,L);
%imshow(B)
%title('Labeled Image')
function [segment, BW] = segment(imageToMask)
    L = superpixels(imageToMask,500);
    
    w = 55;
    h = 500;
    
    bw = 200;
    bh = 50;
    
    sw = size(imageToMask,2);
    sh = size(imageToMask,1);
    
    f = drawrectangle(gca,'Position',[sw/2-w/2 sh/2-h/2 w h],'Color','g');
    foreground = createMask(f,imageToMask);

    b1 = drawrectangle(gca,'Position',[0 0 sw bh],'Color','r');
    b2 = drawrectangle(gca,'Position',[0 sh-bh sw bh],'Color','r');
    b3 = drawrectangle(gca,'Position',[0 0 bw sh],'Color','r');
    b4 = drawrectangle(gca,'Position',[sw-bw 0 bw sh],'Color','r');
    
    background = createMask(b1,imageToMask) + createMask(b2,imageToMask) + createMask(b3,imageToMask) + createMask(b4,imageToMask);
    BW = lazysnapping(imageToMask,L,foreground,background);
    imshow(labeloverlay(imageToMask,BW,'Colormap',[0 1 0]));
    
    segment = bsxfun(@times, imageToMask, cast(BW, 'like', imageToMask));
end
