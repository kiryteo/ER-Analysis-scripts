import math

def eig_hess_2d(dxx, dxy, dyy):
    tmp = math.sqrt((dxx - dyy)^2 + 4*(dxy)^2)

    mu1 = 0.5*(dxx + dxy + tmp)
    mu2 = 0.5*(dxx + dyy - tmp)

    check = (math.abs(mu1) > math.abs(mu2))

    l1 = mu1
    l1(check) = mu2(check)
    l2 = mu2
    l2(check) = mu1(check)


