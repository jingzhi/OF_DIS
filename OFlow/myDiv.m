function div_F = myDiv(F)
%compute grad of 2D binary vectory field
u=F(:,:,1);
v=F(:,:,2);
[H,W]=size(u);
ux=filter2([-1,1],u,'valid');%diff(u,1,2);
vy=filter2([-1;1],v,'valid');%diff(v,1,1);
div_F=ux(1:H-1,1:W-1)+vy(1:H-1,1:W-1);
end

