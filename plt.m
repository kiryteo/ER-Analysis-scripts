r1=-5+10*rand(100,1);
subplot(3,1,1)
plot(r1,'linew',3)
hold on;
a1=conv(r1,[0.1 0.2 0.4 0.2 0.1 ],'same')
plot(a1,'linew',3)
m1=mean(a1)
hold on;
plot(repmat(m1,[100,1]),'linew',3)
gm1=2*(a1>m1)-1;
plot(gm1,'linew',3)
%%%
r2=-5+10*rand(100,1);
subplot(3,1,2)
plot(r2,'linew',3)
hold on;
a2=conv(r2,[0.1 0.2 0.4 0.2 0.1 ],'same')
plot(a2,'linew',3)
m2=mean(a2)
hold on;
plot(repmat(m2,[100,1]),'linew',3)
gm2=2*(a2>m2)-1;
plot(gm2,'linew',3)
%%
subplot(3,1,3)
xc=xcorr(a1,a2);
plot(xc)
