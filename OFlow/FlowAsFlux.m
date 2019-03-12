%%Flow As Flux
close all
clear all
frame1=imread("frame_0021.png");
frame2=imread("frame_0022.png");
occl=imread("frame_0021_occ.png");
[h,w,c]=size(frame1);
x=1:w;
y=1:h;
LK = opticalFlowLK;
frame1Gray=rgb2gray(frame1);
frame2Gray=rgb2gray(frame2);
frame1hsv=rgb2hsv(frame1);
imgStr=edge(frame1hsv(:,:,1),'sobel');
%imshow(imgStr)

%flow=estimateFlow(LK,frame1Gray);
%flow=estimateFlow(LK,frame2Gray);
%u=flow.Vx;%scalar
%v=flow.Vy;%%scalar

flow=readFlowFile("frame_0021_raw.flo");
u=flow(:,:,1);
v=flow(:,:,2);
ux=conv2(u,[-1,1],'valid');
ux=ux(1:h-1,1:w-1);%diff(u,1,2);
vy=conv2(v,[-1;1],'valid');
vy=vy(1:h-1,1:w-1);%diff(v,1,1);
grad_u=myGrad(u);%vec
grad_v=myGrad(v);%vec
div_flow=myDiv(cat(3,u,v));
div_grad_u=myDiv(grad_u);
div_grad_v=myDiv(grad_v);
curl_flow=myCurl(cat(3,u,v));

thres=0.5;
div_flow_po=zeros(size(div_flow));
div_flow_ne=zeros(size(div_flow));
div_flow_po(div_flow>=thres)=1;
div_flow_ne(div_flow<=-thres)=1;
%div_flow_str=edge(div_flow,"canny");
figure
subplot(2,1,1)
imshow(div_flow_po)
title("positive div flow")
subplot(2,1,2)
imshow(div_flow_ne)
title("negative div flow")



% figure
% opflow = opticalFlow(u,v);
% plot(opflow,'DecimationFactor',[10 10],'ScaleFactor',3);
% set(gca, 'YDir','reverse')
% axis([1 w 1 h])
% title("flow")
% %hold on
% figure
% % subplot(2,3,1);
%  quiver(u,v)
%  set(gca, 'YDir','reverse')
%  axis([1 w 1 h])
%  title("flow viz")
% % subplot(2,3,2);
%  figure
%  quiver(grad_u(:,:,1),grad_u(:,:,2));
%  set(gca, 'YDir','reverse')
%  axis([1 w 1 h])
%  title("grad u")
% % subplot(2,3,3);
%  figure
%  quiver(grad_v(:,:,1),grad_v(:,:,2));
%  set(gca, 'YDir','reverse')
%  axis([1 w 1 h])
%  title("grad v")
% % subplot(2,3,4);
 figure
 s=surf(div_flow);
 s.EdgeColor = 'none';
 set(gca, 'YDir','reverse')
 title("div flow")
% % %axis([1 w 1 h])
%  figure
%  s=surf(curl_flow);
%  s.EdgeColor="none";
%  set(gca,'YDir','reverse')
%  title("curl flow")
% % subplot(2,3,5);
%  figure
%  s=surf(div_grad_u);
%   s.EdgeColor = 'none';
%  set(gca, 'YDir','reverse')
% % %axis([1 w 1 h])
%  title("div grad u")
% % subplot(2,3,6);
%  figure
%  s=surf(div_grad_v);
%  s.EdgeColor = 'none';
%  set(gca, 'YDir','reverse')
% % %axis([1 w 1 h])
%  title("div grad v")



