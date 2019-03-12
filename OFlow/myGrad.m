function grad_F = myGrad(F)
%compute grad of 2D scalar field
[H,W]=size(F);
grad_F=zeros(H-1,W-1,2);
Fx=filter2([-1,1],F,'valid');%diff(v,1,1);
Fy=filter2([-1;1],F,'valid');%diff(v,1,1);
%Fx=diff(F,1,2);
%Fy=diff(F,1,1);
grad_F(:,:,1)=Fx(1:H-1,1:W-1);
grad_F(:,:,2)=Fy(1:H-1,1:W-1);
end

