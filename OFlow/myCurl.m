function curl_F = myCurl(F)
%compute grad of 2D scalar field;
u=F(:,:,1);
v=F(:,:,2);
[H,W]=size(u);
%curl_F=zeros(H-2,W-2);
vx=filter2([-1,1],v,'valid');%diff(v,1,1);
uy=filter2([-1;1],u,'valid');%diff(v,1,1);
%uy=diff(u,1,1);
%vx=diff(v,1,2);
curl_F=vx(1:H-1,1:W-1)-uy(1:H-1,1:W-1);
end

