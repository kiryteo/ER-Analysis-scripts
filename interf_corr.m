function [output,IFC]=interframe_corr(im_1,im_2)
old=double(im_1);
new=double(im_2);
%Calculate the net sum of all pixels
sum_old=sum(sum(old));sum_new=sum(sum(new));
[rows,cols]=size(old);
num1=0;num2=0;den1=0;den2=0; %initialisation
for i=1:rows
    for j=1:cols
        den1=den1+(old(i,j)-sum_old)^2; %denominator factor 1
        den2=den2+(new(i,j)-sum_new)^2; %denominator factor 2
    end
end
D=sqrt(double(den1*den2)); % Net Denominator
for i=1:rows
    for j=1:cols
        num1=(old(i,j)-sum_old); %Numerator factor 1
        num2=(new(i,j)-sum_new); %Numerator factor 1

        IFC(i,j)=num1*num2/D; %Inter Frame Correletion Coeff
    end
end
output=mean2(IFC); % Average of coeffs
end
