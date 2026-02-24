#include <iostream>
#include <vector>
#include <cmath>

// calculates the information given number of individuals and sigma:
double information(double num_individuals, double sigma)
{
  return num_individuals/(2*sigma*sigma);
}

double expected_sample_size(
        std::vector<double> phi,
        std::vector<double> psi,
        std::vector<double> parameters)
{
    double expected_sample_size {0};
    double n_per_stage = parameters.at(0);

    // separating indexing vector with multiplication without loop by 
    // initializing a new variable mult
    double mult = 1;
    for(size_t i = 0; i < phi.size(); i++)
    {
        // based on expected sample size equation, suspect that:
        // phi is the probability of stopping for efficacy and
        // psi is the probability of stopping for futility 
        expected_sample_size += mult * n_per_stage * (phi.at(i)+psi.at(i));
        mult++;
    }

    return expected_sample_size;
}

// finds normal pdf, cdf and inverse cdf
double normal_pdf(double z)
{
    return ((1.)/sqrt(2*M_PI)) * exp((-z*z)/2);
}

// Abramowitz & Stegun (1964) approximation 
// of the standard normal cdf accurate to < 7.5e-8 based on
// Abramowitz/Stegun 26.2.17
// maybe originally in  Approximations for Digital Computers
// Cecil Hastings 1955
double normal_cdf(double z)
{
    if (z > 6.0)
    {
        return 1.0;
    }
  
    if (z < -6.0)
    {
        return 0.0;
    }

    double b1 {0.31938153};
    double b2 {-0.356563782};
    double b3 {1.781477937};
    double b4 {-1.821255978};
    double b5 {1.330274429};

    double p {0.2316419};

    // numerical approximation for 1/sqrt(2*pi)
    double c2 {0.3989423};

    double a = fabs(z);
    double t = 1.0/(1.0+a*p);

    // the below calculation is equivalent to:
    // 1 - 1/sqrt(2*pi) * exp( -(pow(z,2)/2) ) * ( b1*t + b2*pow(t,2.0) 
    //                                             + b3*pow(t,3.0) + b4*pow(t,4.0) 
    //                                             + b5*pow(t,5.0) )
    double b = c2*exp((-z)*(z/2.0));
    double n = ((((b5*t+b4)*t+b3)*t+b2)*t+b1)*t;

    n = 1.0 - b*n;

    if (z < 0.0) 
    {
        n = 1.0 - n;
    }

    return n;
}

void converthtopsi(
        std::vector<std::vector<double>>& h,
        std::vector<std::vector<double>>& z,
        std::vector<double>& psi,
        std::vector<double>& parameters,
        double delta0,
        double new_delta,
        double sigma)
{
    double i{};
    size_t j{};

    psi.clear();

    psi.push_back(
        normal_cdf(parameters.at(1)
                   - new_delta * sqrt(parameters.at(0))/sqrt(2*sigma*sigma))
    );

    for (i = 1; i <= static_cast<double>(z.size()); i++)
    {
        psi.push_back(0);
    
        for (j = 0; 
             j < z.at(static_cast<size_t>(i)-1).size(); 
             j++)
        {
            // this calculation is the same as the below in convert to phi with
            // two minor differences
            psi.at(static_cast<size_t>(i))
                // equation ell(z_k, I_k | theta_2, theta_1) near equation 19.13    
                += exp((new_delta-delta0)*z.at(static_cast<size_t>(i)-1).at(j)*sqrt(information(i*parameters.at(0),sigma))
                - (pow(new_delta,2)-pow(delta0,2))
                * information(i*parameters.at(0),sigma)/2)
                // equation h_k-1(i_k-1 | theta1)
                * (h.at(static_cast<size_t>(i)-1).at(j)
                // this is e_k-1(z_k-1(i_k-1)), b_k | theta2)
                // except 1- is at the front
                * (1 - normal_cdf(
                    (z.at(static_cast<size_t>(i)-1).at(j)*sqrt(information(i*parameters.at(0),sigma))
                    // term 1 of eqution 19.6 except we are indexing the odd value
                    // here we have (i+1)*2-1 rather than (i+1)*2
                    - parameters.at(static_cast<size_t>((i+1)*2-1))*sqrt(information((i+1)*parameters.at(0),sigma))
                    // term 2 of equation 19.6
                    + new_delta * (information((i+1)*parameters.at(0),sigma)
                        - information(i*parameters.at(0),sigma)))
                    // divided by sqrt(delta_k) in equation 19.6 
                    / (sqrt(information((i+1)*parameters.at(0),sigma) - information(i*parameters.at(0),sigma))))));
        }
    
    }

}

void converthtophi(
        std::vector<std::vector<double>>& h,
        std::vector<std::vector<double>>& z,
        std::vector<double>& phi,
        std::vector<double>& parameters,
        double delta0,
        double new_delta,
        double sigma)
{
    double i;
    size_t j;

    phi.clear();

    phi.push_back(
        1 - normal_cdf(parameters.at(2)
            - new_delta*sqrt(parameters.at(0))/sqrt(2*sigma*sigma))
    );

    for(i = 1; i <= static_cast<double>(z.size()); i++)
    {
        phi.push_back(0);
    
        for(j = 0; j < z.at(static_cast<size_t>(i)-1).size(); j++)
        {
            phi.at(static_cast<size_t>(i)) 
                // equation ell(z_k, I_k | theta_2, theta_1) near equation 19.13    
                += exp((new_delta-delta0)*z.at(static_cast<size_t>(i)-1).at(j)*sqrt(information(i*parameters.at(0),sigma))
                - (pow(new_delta,2)-pow(delta0,2))
                * information(i*parameters.at(0),sigma)/2)
                // equation h_k-1(i_k-1 | theta1)
                * (h.at(static_cast<size_t>(i)-1).at(j)
                // e_k-1(z_k-1(i_k-1)), b_k | theta2)
                // equation 19.6
                * (normal_cdf(
                    // term 1 of equation 19.6
                    (z.at(static_cast<size_t>(i)-1).at(j) * sqrt(information(i*parameters.at(0),sigma))
                    // term 3 of equation 19.6
                    - parameters.at(static_cast<size_t>((i+1)*2))*sqrt(information((i+1)*parameters.at(0),sigma))
                    // term 2 of equation 19.6
                    + new_delta*(information((i+1)*parameters.at(0),sigma)
                        - information(i*parameters.at(0),sigma)))
                    // divided by sqrt(delta_k) in equation 19.6
                    / (sqrt(information((i+1)*parameters.at(0),sigma) - information(i*parameters.at(0),sigma))))));
        }
    
    }

}

// trialproperties_seq uses the method given in Section 19.2 of Jennison and 
// Turnbull (2000) to find the probability of stopping at each stage in a 
// sequential trial using Z-tests
void trialproperties_seq(
    std::vector<std::vector<double>>& h,
    std::vector<std::vector<double>>& z,
    // expecting parameters with order sample size at each stage, lower1, upper1,
    // lower2, upper2, ... lower_n, upper_n
    // triangular boudaries {47, 3.13269e-16, 2.13029, 1.12976, 1.88293, 1.84489, 1.84489} 
    std::vector<double> parameters = {47, 3.13269e-16, 2.13029, 1.12976, 1.88293, 1.84489, 1.84489},
    double delta0 = 0,
    double delta1 = 1,
    double sigma = 3,
    double typeIerror = 0.05,
    double power = 0.9
)
{

    // Function will find typeIerror and power for trial parameters. If 
    // checkdm==1, the worst-case scenario delta will be found together with its 
    // expected sample size. Else, both will be returned as 0

    size_t i{};
    double j{};

    // this temporary vector will be used throughout all of the below for loops
    // as a placeholder for the actual vector being calculated
    std::vector<double> tempvector;

    // get grid of points to use
    // the grid is x_i with elements {x_1, . . . , x_6r-1} (see immediately 
    // below equation 19.9)
    // r is defined as 16 in the code below
    // Also note that x is a 2 dimensional vector, that can have different 
    // lengths for each dimension, for example:
    // [[1, 2, 3],
    //  [4, 5, 6, 7, 8]]
    // x has the same number of rows as boundary pairs minus 1
    std::vector<std::vector<double>> x;

    // loop through the boudaries
    // parameters.size is one larger than 2*stages because it also contains
    // the sample size within it. Also, subtract 1 because the last boundary
    // is equal in one-stage designs, hence (parameters.size-1)/2-1
    for (i = 0; i < (parameters.size()-1)/2 - 1; i++)
    {
        tempvector.clear();

        // add the first boundary value to the grid of points
        tempvector.push_back(parameters.at(i*2+1));
        
        // initialize mean at each stage
        // mean_at_stage is theta * sqrt(information at each stage)
        double mean_at_stage  = delta0 * sqrt(
            information((static_cast<double>(i)+1)*parameters.at(0),sigma)
        );
        
        // these are the first 15 grid points
        for (j = 1; j <= 15; j++)
        {
            // the following if statements check to see if the value of the 
            // next grid point is between the two boundary values of interest
            // if not, it would be invalid and therefore is not included in
            // the grid
            if (mean_at_stage - (3+4*log(16.0/j)) < parameters.at(i*2+2) 
                && mean_at_stage - (3+4*log(16.0/j)) > parameters.at(i*2+1))
            {
                tempvector.push_back(mean_at_stage-(3+4*log(16.0/j)));
            }
        }
        
        // these are the next 79 grid points
        for (j = 16; j <= 5*16; j++)
        {
            if (mean_at_stage-(3.0-3*(j-16.0)/(2*16.0)) < parameters.at(i*2+2) 
                && mean_at_stage-(3.0-3*(j-16.0)/(2*16.0)) > parameters.at(i*2+1))
            {
                tempvector.push_back(mean_at_stage-(3.0-3*(j-16.0)/(2*16.0)));
            }
        }
        
        // these are the last 15 grid points
        for (j = 5*16+1; j <= 6*16-1; j++)
        {
            if (mean_at_stage+(3.0+4*log(16.0/(6*16-j))) < parameters.at(i*2+2) 
                && mean_at_stage+(3.0+4*log(16.0/(6*16-j)))>parameters.at(i*2+1))
            {
                tempvector.push_back(mean_at_stage+(3.0+4*log(16.0/(6*16-j))));
            }
        }
        
        // add the last boundary value to the grid of points
        tempvector.push_back(parameters.at(i*2+2));

        // fill our grid with the temporary vector
        x.push_back(tempvector);
    }
    
    // to integrate from -inf to inf, 12r-3 grid points are used. Recall r is 
    // defined as 16 above. The odd numbered grid points were defined above in
    // the vector x. The final z vector will include the even numbered grid
    // points as the midpoints. For example, if x = {1, 2, 3}, then temp would
    // be {1.5, 2.5} and the full z vector {1, 1.5, 2, 2.5, 3}.
    
    for (i = 0; i < x.size(); i++)
    {
        tempvector.clear();
        
        for (size_t j = 0; j < x.at(i).size()-1; j++)
        {
            tempvector.push_back(x.at(i).at(j));
            tempvector.push_back((x.at(i).at(j)+x.at(i).at(j+1))/2);
        }
        
        tempvector.push_back(x.at(i).at(x.at(i).size()-1));
        z.push_back(tempvector);

    }
    
    // the weights are calculated based on equation 19.10
    // quadrature approximation is used based on Simpson's rule
    std::vector<std::vector<double>> weights;
    
    for (i = 0; i < z.size();  i++)
    {
        // the first weight
        tempvector.clear();
        tempvector.push_back((z.at(i).at(2)-z.at(i).at(0))/6);
        
        for (size_t j = 2; j <= z.at(i).size() - 1; j++)
        {
            // for even numbered weights after the first weight
            if (j % 2 == 0)
            {
                tempvector.push_back(4.0*(z.at(i).at(j)-z.at(i).at(j-2))/6);
            }
            
            // for odd numbered weights after the first weight
            else
            {
                tempvector.push_back((z.at(i).at(j+1)-z.at(i).at(j-3))/6);
            }
        }
        
        // the last weight
        tempvector.push_back((z.at(i).at(z.at(i).size()-1)-z.at(i).at(z.at(i).size()-3))/6);
        weights.push_back(tempvector);

    }
    
    // h is the vector of k-1 dimensions (k = num stages) that collects the 
    // elements of a sum of for quick calculation of integrals between the 
    // z grid points defined above
    
    size_t k;
    
    for (i = 0; i < z.size(); i++)
    {
        tempvector.clear();
        
        double info_k = information((static_cast<double>(i)+1)*parameters.at(0),sigma); 
        double info_kminus1 =information(static_cast<double>(i)*parameters.at(0),sigma);
        double delta_k = info_k - info_kminus1;

        if (i == 0)
        {
            for (size_t j = 0; j < z.at(i).size(); j++)
            {
                // f_1(z_1 | sigma) just before equation 19.4
                tempvector.push_back(
                    weights.at(i).at(j)*normal_pdf(z.at(i).at(j)-delta0*sqrt(info_k))
                );
            }
        }
        
        else
        {
            for (size_t j = 0; j < z.at(i).size(); j++)
            {
                tempvector.push_back(0);
                
                for (k = 0; k < z.at(i-1).size(); k++)
                {
                    tempvector.at(j) += 
                        // h_k-1(i_k-1 | theta)
                        h.at(i-1).at(k)
                        // w_k(i_k)
                        * weights.at(i).at(j)
                        // equation 19.4
                        * (sqrt(info_k)/sqrt(delta_k))
                        * normal_pdf(
                            // term 1
                            (z.at(i).at(j)*sqrt(info_k)
                            // term 2
                            - z.at(i-1).at(k)*sqrt(info_kminus1)
                            // term 3
                            - delta0*(delta_k))
                            // divided by
                            / (sqrt(delta_k))
                        );
                }
            }
        }
        
        h.push_back(tempvector);

    }
    
    // phi gives the probability of stopping for efficacy at each stage
    std::vector<double> phi;
    std::vector<double> psi;

    // this is being used to calculate the probabilities from the h vectors
    // it is taking two of the same delta values because its using the trick
    // discussed after equation 19.13
    converthtophi(h, z, phi, parameters, delta0, delta0, sigma);

    //calculate typeIerror by summing up phi:
    typeIerror = 0;
    
    for(i = 0; i < phi.size(); i++)
    {
        typeIerror += phi.at(i);
    }
    
    converthtopsi(h, z, psi, parameters, delta0, delta0, sigma);
    
    double expected_sample_size_null{};
    expected_sample_size_null = expected_sample_size(phi, psi, parameters);
    
    converthtophi(h, z, phi, parameters, delta0, delta1, sigma);
    
    power = 0;
    
    for(i = 0; i < phi.size(); i++)
    {
        power += phi.at(i);
    }

    converthtopsi(h, z, psi, parameters, delta0, delta1, sigma);
    
    double expected_sample_size_crd{};
    expected_sample_size_crd = expected_sample_size(phi, psi, parameters);

    std::cout << "This is h: " << "\n";
    std::cout << "h has " << h.size() << " rows." << "\n";
    for (size_t i = 0; i < h.size(); i++)
    {
        std::cout << "Row " << i << " has " << h[i].size() << " columns." << "\n";
    }
    std::cout << "\n";
    for (size_t i = 0; i < h.size(); i++)
    {
        std::cout << "This is row " << i << "\n";
        for (size_t j = 0; j < h[i].size(); j++)
        {
            std::cout << h[i][j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n\n";

    std::cout << "This is the expected sample size under the null: ";
    std::cout << expected_sample_size_null << "\n";
    std::cout << "This is the expected sample size under the alternative: ";
    std::cout << expected_sample_size_crd << "\n\n";

}

// finds the delta which gives highest expected sample size for a given design
void finddeltaminimax_seq(
        std::vector<std::vector<double>>& h,
        std::vector<std::vector<double>>& z,
        std::vector<double>& parameters,
        double delta0,
        double new_delta,
        double sigma,
        double& deltaminimax,
        double& maxen)
{

    // function values to use for run

    double lowerdelta=delta0;
    double middelta=new_delta;
    double upperdelta=delta0+2*(new_delta-delta0);
    double tempdelta;
    double loweren;
    double miden;
    double upperen;
    double tempen;

    std::vector<double> lowerphi;
    std::vector<double> lowerpsi;
    std::vector<double> midphi;
    std::vector<double> midpsi;
    std::vector<double> upperphi;
    std::vector<double> upperpsi;
    std::vector<double> tempphi;
    std::vector<double> temppsi;

    converthtophi(h,z,lowerphi,parameters,delta0,lowerdelta,sigma);
    converthtophi(h,z,midphi,parameters,delta0,middelta,sigma);
    converthtophi(h,z,upperphi,parameters,delta0,upperdelta,sigma);
    converthtopsi(h,z,lowerpsi,parameters,delta0,lowerdelta,sigma);
    converthtopsi(h,z,midpsi,parameters,delta0,middelta,sigma);
    converthtopsi(h,z,upperpsi,parameters,delta0,upperdelta,sigma);

    loweren = expected_sample_size(lowerphi,lowerpsi,parameters);
    miden = expected_sample_size(midphi,midpsi,parameters); 
    upperen = expected_sample_size(upperphi,upperpsi,parameters);

    //check whether miden is higher than both loweren and upperen
    
    while (miden<loweren || miden<upperen)
    {
        lowerdelta-=(new_delta-delta0);
        upperdelta+=(new_delta-delta0);
        converthtophi(h,z,lowerphi,parameters,delta0,lowerdelta,sigma);
        converthtophi(h,z,upperphi,parameters,delta0,upperdelta,sigma);
        converthtopsi(h,z,lowerpsi,parameters,delta0,lowerdelta,sigma);
        converthtopsi(h,z,upperpsi,parameters,delta0,upperdelta,sigma);
        loweren=expected_sample_size(lowerphi,lowerpsi,parameters);
        upperen=expected_sample_size(upperphi,upperpsi,parameters);
    }
    
    //find delta which gives maximum expected sample size:
    while((middelta-lowerdelta)>1e-3 || (upperdelta-middelta)>1e-3) 
    {
        if ((upperdelta-middelta)>(middelta-lowerdelta))
        {
            tempdelta=(upperdelta+middelta)/2;
            converthtophi(h,z,tempphi,parameters,delta0,tempdelta,sigma);
            converthtopsi(h,z,temppsi,parameters,delta0,tempdelta,sigma);
            tempen=expected_sample_size(tempphi,temppsi,parameters);
            
            if (tempen<miden)
            {
                upperen=tempen;
                upperdelta=tempdelta;
                upperphi=tempphi;
                upperpsi=temppsi;
            }
            
            else
            {
                loweren=miden;
                lowerdelta=middelta;
                lowerphi=midphi;
                lowerpsi=midpsi;
                miden=tempen;
                middelta=tempdelta;
                midphi=tempphi;
                midpsi=temppsi;
            }
        }
        
        else
        {
            tempdelta=(lowerdelta+middelta)/2;
            converthtophi(h,z,tempphi,parameters,delta0,tempdelta,sigma);
            converthtopsi(h,z,temppsi,parameters,delta0,tempdelta,sigma);
            tempen=expected_sample_size(tempphi,temppsi,parameters);
            
            if(tempen<miden)
            {
                loweren=tempen;
                lowerdelta=tempdelta;
                lowerphi=tempphi;
                lowerpsi=temppsi;
            }
            
            else
            {
                upperen=miden;
                upperdelta=middelta;
                upperphi=midphi;
                upperpsi=midpsi;
                miden=tempen;
                middelta=tempdelta;
                midphi=tempphi;
                midpsi=temppsi;
            }
        }
    }
    
    deltaminimax=middelta;
    maxen=miden;
    
    std::cout << "This is the delta with the highest sample size: ";
    std::cout << deltaminimax << "\n";
    std::cout << "This is the maximum expected sample size: ";
    std::cout << maxen;
    std::cout << "\n\n";
}

// finds the delta which gives highest expected sample size for a given design
void new_finddeltaminimax_seq(
        std::vector<std::vector<double>>& h,
        std::vector<std::vector<double>>& z,
        std::vector<double>& parameters,
        double delta0,
        double delta1,
        double sigma)
{

    double epsilon {1e-4};
    
    double lowerdelta = delta0;
    double upperdelta = delta1 * 10.; 
    
    std::vector<double> lowerphi;
    std::vector<double> upperphi;
    
    std::vector<double> lowerpsi;
    std::vector<double> upperpsi;
    
    double lower_ess = expected_sample_size(lowerphi, lowerpsi, parameters);
    double upper_ess = expected_sample_size(upperphi, upperpsi, parameters); 

    //find delta which gives maximum expected sample size:
    while( std::abs(lowerdelta - upperdelta) > epsilon ) 
    {

        converthtophi(h, z, lowerphi, parameters, delta0, lowerdelta, sigma);
        converthtopsi(h, z, lowerpsi, parameters, delta0, lowerdelta, sigma);

        converthtophi(h, z, upperphi, parameters, delta0, upperdelta, sigma);
        converthtopsi(h, z, upperpsi, parameters, delta0, upperdelta, sigma);

        lower_ess = expected_sample_size(lowerphi, lowerpsi, parameters);
        upper_ess = expected_sample_size(upperphi, upperpsi, parameters);
        
        if (upper_ess > lower_ess)
        {
            lowerdelta = (lowerdelta + upperdelta) / 2;
        }

        else
        {
            upperdelta = (lowerdelta + upperdelta) / 2;
        }

    }
    
    std::cout << "***** NEW FUNCTION *****" << "\n";
    std::cout << "This is the delta with the highest sample size: ";
    std::cout << upperdelta << "\n";
    std::cout << "This is the maximum expected sample size: ";
    std::cout << upper_ess;
    std::cout << "\n\n";
}

int main()
{

    std::vector<std::vector<double>> h {};
    std::vector<std::vector<double>> z {};
    std::vector<double> parameters {};
    double delta0 {0.};
    double delta1 {1.};
    double sigma {3.};
    double typeIerror {0.05};
    double power {0.9};

    trialproperties_seq(
        h,
        z,
        parameters = {47, 3.13269e-16, 2.13029, 1.12976, 1.88293, 1.84489, 1.84489},
        delta0,
        delta1,
        sigma,
        typeIerror,
        power
    );

    double delta_minimax {};
    double max_en {};

    finddeltaminimax_seq(
        h,
        z,
        parameters,
        delta0,
        delta1,
        sigma,
        delta_minimax,
        max_en
    );

    new_finddeltaminimax_seq(
        h,
        z,
        parameters,
        delta0,
        delta1,
        sigma
    );

    return 0;
}
