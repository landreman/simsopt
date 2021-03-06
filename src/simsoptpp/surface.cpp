#include "surface.h"

template<class Array>
void Surface<Array>::least_squares_fit(Array& target_values) {
    if(target_values.shape(0) != numquadpoints_phi)
        throw std::runtime_error("Wrong first dimension for target_values. Should match numquadpoints_phi.");
    if(target_values.shape(1) != numquadpoints_theta)
        throw std::runtime_error("Wrong second dimension for target_values. Should match numquadpoints_theta.");
    if(target_values.shape(2) != 3)
        throw std::runtime_error("Wrong third dimension for target_values. Should be 3.");

    if(!qr){
        auto dg_dc = this->dgamma_by_dcoeff();
        Eigen::MatrixXd A = Eigen::MatrixXd(numquadpoints_phi*numquadpoints_theta*3, num_dofs());
        int counter = 0;
        for (int i = 0; i < numquadpoints_phi; ++i) {
            for (int j = 0; j < numquadpoints_theta; ++j) {
                for (int d = 0; d < 3; ++d) {
                    for (int c = 0; c  < num_dofs(); ++c ) {
                        A(counter, c) = dg_dc(i, j, d, c);
                    }
                    counter++;
                }
            }
        }
        qr = std::make_unique<Eigen::FullPivHouseholderQR<Eigen::MatrixXd>>(A.fullPivHouseholderQr());
    }
    Eigen::VectorXd b = Eigen::VectorXd(numquadpoints_phi*numquadpoints_theta*3);
    int counter = 0;
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            for (int d = 0; d < 3; ++d) {
                b(counter++) = target_values(i, j, d);
            }
        }
    }
    Eigen::VectorXd x = qr->solve(b);
    vector<double> dofs(x.data(), x.data() + x.size());
    this->set_dofs(dofs);
}

template<class Array>
void Surface<Array>::fit_to_curve(Curve<Array>& curve, double radius, bool flip_theta) {
    Array curvexyz = xt::zeros<double>({numquadpoints_phi, 3});
    curve.gamma_impl(curvexyz, quadpoints_phi);
    Array target_values = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
    for (int i = 0; i < numquadpoints_phi; ++i) {
        double phi = 2*M_PI*quadpoints_phi[i];
        double R = sqrt(pow(curvexyz(i, 0), 2) + pow(curvexyz(i, 1), 2));
        double z = curvexyz(i, 2);
        for (int j = 0; j < numquadpoints_theta; ++j) {
            double theta = 2*M_PI*quadpoints_theta[j];
            if(flip_theta)
                theta *= -1;
            target_values(i, j, 0) = (R + radius * cos(theta))*cos(phi);
            target_values(i, j, 1) = (R + radius * cos(theta))*sin(phi);
            target_values(i, j, 2) = z + radius * sin(theta);
        }
    }
    this->least_squares_fit(target_values);
}

template<class Array>
void Surface<Array>::scale(double scale) {
    Array target_values = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
    auto gamma = this->gamma();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        double phi = 2*M_PI*quadpoints_phi[i];
        double meanx = 0.;
        double meany = 0.;
        double meanz = 0.;
        for (int j = 0; j < numquadpoints_theta; ++j) {
            meanx += gamma(i, j, 0);
            meany += gamma(i, j, 1);
            meanz += gamma(i, j, 2);
        }
        meanx *= 1./numquadpoints_theta;
        meany *= 1./numquadpoints_theta;
        meanz *= 1./numquadpoints_theta;
        for (int j = 0; j < numquadpoints_theta; ++j) {
            target_values(i, j, 0) = meanx + scale * (gamma(i, j, 0) - meanx);
            target_values(i, j, 1) = meany + scale * (gamma(i, j, 1) - meany);
            target_values(i, j, 2) = meanz + scale * (gamma(i, j, 2) - meanz);
        }
    }
    this->least_squares_fit(target_values);
}

template<class Array>
void Surface<Array>::extend_via_normal(double scale) {
    Array target_values = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
    auto gamma = this->gamma();
    auto n = this->normal();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            auto nij_norm = sqrt(n(i, j, 0)*n(i, j, 0) + n(i, j, 1)*n(i, j, 1) + n(i, j, 2)*n(i, j, 2));
            target_values(i, j, 0) = gamma(i, j, 0) + scale * n(i, j, 0) / nij_norm;
            target_values(i, j, 1) = gamma(i, j, 1) + scale * n(i, j, 1) / nij_norm;
            target_values(i, j, 2) = gamma(i, j, 2) + scale * n(i, j, 2) / nij_norm;
        }
    }
    this->least_squares_fit(target_values);
}

template<class Array>
void Surface<Array>::normal_impl(Array& data)  { 
    auto dg1 = this->gammadash1();
    auto dg2 = this->gammadash2();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            data(i, j, 0) = dg1(i, j, 1)*dg2(i, j, 2) - dg1(i, j, 2)*dg2(i, j, 1);
            data(i, j, 1) = dg1(i, j, 2)*dg2(i, j, 0) - dg1(i, j, 0)*dg2(i, j, 2);
            data(i, j, 2) = dg1(i, j, 0)*dg2(i, j, 1) - dg1(i, j, 1)*dg2(i, j, 0);
        }
    }
};
template<class Array>
void Surface<Array>::dnormal_by_dcoeff_impl(Array& data)  { 
    auto dg1 = this->gammadash1();
    auto dg2 = this->gammadash2();
    auto dg1_dc = this->dgammadash1_by_dcoeff();
    auto dg2_dc = this->dgammadash2_by_dcoeff();
    int ndofs = num_dofs();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            for (int m = 0; m < ndofs; ++m ) {
                data(i, j, 0, m) =  dg1_dc(i, j, 1, m)*dg2(i, j, 2) - dg1_dc(i, j, 2, m)*dg2(i, j, 1);
                data(i, j, 0, m) += dg1(i, j, 1)*dg2_dc(i, j, 2, m) - dg1(i, j, 2)*dg2_dc(i, j, 1, m);
                data(i, j, 1, m) =  dg1_dc(i, j, 2, m)*dg2(i, j, 0) - dg1_dc(i, j, 0, m)*dg2(i, j, 2);
                data(i, j, 1, m) += dg1(i, j, 2)*dg2_dc(i, j, 0, m) - dg1(i, j, 0)*dg2_dc(i, j, 2, m);
                data(i, j, 2, m) =  dg1_dc(i, j, 0, m)*dg2(i, j, 1) - dg1_dc(i, j, 1, m)*dg2(i, j, 0);
                data(i, j, 2, m) += dg1(i, j, 0)*dg2_dc(i, j, 1, m) - dg1(i, j, 1)*dg2_dc(i, j, 0, m);
            }
        }
    }
};
template<class Array>
void Surface<Array>::d2normal_by_dcoeffdcoeff_impl(Array& data)  { 
    auto dg1 = this->gammadash1();
    auto dg2 = this->gammadash2();
    auto dg1_dc = this->dgammadash1_by_dcoeff();
    auto dg2_dc = this->dgammadash2_by_dcoeff();
    int ndofs = num_dofs();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            for (int m = 0; m < ndofs; ++m ) {
                for (int n = 0; n < ndofs; ++n ) {
                    data(i, j, 0, m, n) =  dg1_dc(i, j, 1, m)*dg2_dc(i, j, 2, n) - dg1_dc(i, j, 2, m)*dg2_dc(i, j, 1, n);
                    data(i, j, 0, m, n) += dg1_dc(i, j, 1, n)*dg2_dc(i, j, 2, m) - dg1_dc(i, j, 2, n)*dg2_dc(i, j, 1, m);
                    data(i, j, 1, m, n) =  dg1_dc(i, j, 2, m)*dg2_dc(i, j, 0, n) - dg1_dc(i, j, 0, m)*dg2_dc(i, j, 2, n);
                    data(i, j, 1, m, n) += dg1_dc(i, j, 2, n)*dg2_dc(i, j, 0, m) - dg1_dc(i, j, 0, n)*dg2_dc(i, j, 2, m);
                    data(i, j, 2, m, n) =  dg1_dc(i, j, 0, m)*dg2_dc(i, j, 1, n) - dg1_dc(i, j, 1, m)*dg2_dc(i, j, 0, n);
                    data(i, j, 2, m, n) += dg1_dc(i, j, 0, n)*dg2_dc(i, j, 1, m) - dg1_dc(i, j, 1, n)*dg2_dc(i, j, 0, m);
                }
            }
        }
    }
};


template<class Array>
double Surface<Array>::area() {
    double area = 0.;
    auto n = this->normal();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            area += sqrt(n(i,j,0)*n(i,j,0) + n(i,j,1)*n(i,j,1) + n(i,j,2)*n(i,j,2));
        }
    }
    return area/(numquadpoints_phi*numquadpoints_theta);
}

template<class Array>
void Surface<Array>::darea_by_dcoeff_impl(Array& data) {
    data *= 0.;
    auto n = this->normal();
    auto dn_dc = this->dnormal_by_dcoeff();
    int ndofs = num_dofs();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            for (int m = 0; m < ndofs; ++m) {
                data(m) += (dn_dc(i,j,0,m)*n(i,j,0) + dn_dc(i,j,1,m)*n(i,j,1) + dn_dc(i,j,2,m)*n(i,j,2)) / sqrt(n(i,j,0)*n(i,j,0) + n(i,j,1)*n(i,j,1) + n(i,j,2)*n(i,j,2));
            }
        }
    }
    data *= 1./ (numquadpoints_phi*numquadpoints_theta);
}
template<class Array>
void Surface<Array>::d2area_by_dcoeffdcoeff_impl(Array& data) {
    data *= 0.;
    double norm, dnorm_dcoeffn;
    auto nor = this->normal();
    auto dnor_dc = this->dnormal_by_dcoeff();
    auto d2nor_dcdc = this->d2normal_by_dcoeffdcoeff();
    int ndofs = num_dofs();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            for (int m = 0; m < ndofs; ++m) {
                for (int n = 0; n < ndofs; ++n) {
                    norm = sqrt(nor(i,j,0)*nor(i,j,0)
                            + nor(i,j,1)*nor(i,j,1) 
                            + nor(i,j,2)*nor(i,j,2));
                    dnorm_dcoeffn = (dnor_dc(i,j,0,n)*nor(i,j,0) 
                            + dnor_dc(i,j,1,n)*nor(i,j,1) 
                            + dnor_dc(i,j,2,n)*nor(i,j,2)) / norm;
                    data(m,n) +=  dnor_dc(i,j,0,m) * (dnor_dc(i,j,0,n) * norm - dnorm_dcoeffn * nor(i,j,0)) / (norm*norm)
                        + dnor_dc(i,j,1,m) * (dnor_dc(i,j,1,n) * norm - dnorm_dcoeffn * nor(i,j,1)) / (norm*norm)
                        + dnor_dc(i,j,2,m) * (dnor_dc(i,j,2,n) * norm - dnorm_dcoeffn * nor(i,j,2)) / (norm*norm)
                        + d2nor_dcdc(i,j,0,m,n) * nor(i,j,0) / norm
                        + d2nor_dcdc(i,j,1,m,n) * nor(i,j,1) / norm
                        + d2nor_dcdc(i,j,2,m,n) * nor(i,j,2) / norm;
                }
            }
        }
    }
    data *= 1./ (numquadpoints_phi*numquadpoints_theta);
}


template<class Array>
double Surface<Array>::volume() {
    double volume = 0.;
    auto n = this->normal();
    auto xyz = this->gamma();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            volume += (1./3) * (xyz(i, j, 0)*n(i,j,0)+xyz(i,j,1)*n(i,j,1)+xyz(i,j,2)*n(i,j,2));
        }
    }
    return volume/(numquadpoints_phi*numquadpoints_theta);
}

template<class Array>
void Surface<Array>::dvolume_by_dcoeff_impl(Array& data) {
    data *= 0.;
    auto n = this->normal();
    auto dn_dc = this->dnormal_by_dcoeff();
    auto xyz = this->gamma();
    auto dxyz_dc = this->dgamma_by_dcoeff();
    int ndofs = num_dofs();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            for (int m = 0; m < ndofs; ++m) {
                data(m) += (1./3) * (dxyz_dc(i,j,0,m)*n(i,j,0)+dxyz_dc(i,j,1,m)*n(i,j,1)+dxyz_dc(i,j,2,m)*n(i,j,2));
                data(m) += (1./3) * (xyz(i,j,0)*dn_dc(i,j,0,m)+xyz(i,j,1)*dn_dc(i,j,1,m)+xyz(i,j,2)*dn_dc(i,j,2,m));
            }
        }
    }
    data *= 1./ (numquadpoints_phi*numquadpoints_theta);
}
template<class Array>
void Surface<Array>::d2volume_by_dcoeffdcoeff_impl(Array& data) {
    data *= 0.;
    auto nor = this->normal();
    auto dnor_dc = this->dnormal_by_dcoeff();
    auto d2nor_dcdc = this->d2normal_by_dcoeffdcoeff();
    auto xyz = this->gamma();
    auto dxyz_dc = this->dgamma_by_dcoeff();
    int ndofs = num_dofs();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            for (int m = 0; m < ndofs; ++m){ 
                for (int n = 0; n < ndofs; ++n){ 
                    data(m,n) += (1./3) * (dxyz_dc(i,j,0,m)*dnor_dc(i,j,0,n)
                            +dxyz_dc(i,j,1,m)*dnor_dc(i,j,1,n)
                            +dxyz_dc(i,j,2,m)*dnor_dc(i,j,2,n));
                    data(m,n) += (1./3) * (xyz(i,j,0)*d2nor_dcdc(i,j,0,m,n) + dxyz_dc(i,j,0,n) * dnor_dc(i,j,0,m)
                            +xyz(i,j,1)*d2nor_dcdc(i,j,1,m,n) + dxyz_dc(i,j,1,n) * dnor_dc(i,j,1,m)
                            +xyz(i,j,2)*d2nor_dcdc(i,j,2,m,n) + dxyz_dc(i,j,2,n) * dnor_dc(i,j,2,m));
                }
            }
        }
    }
    data *= 1./ (numquadpoints_phi*numquadpoints_theta);
}

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class Surface<Array>;
