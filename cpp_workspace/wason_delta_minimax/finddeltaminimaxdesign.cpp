#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <random>

// Wichmann–Hill uniform pseudorandom number generator
// Wichmann, Brian A.; Hill, I. David (1982). "Algorithm AS 183: An Efficient 
// and Portable Pseudo-Random Number Generator". Journal of the Royal 
// Statistical Society. Series C (Applied Statistics). 
// this was in the original code and has been replaced by the functions
// get_rng() and uniform_random_01()
// double uniform_random_01()
// {
//     static long ix{1};
//     static long iy{1};
//     static long iz{1};
// 
//     double r{};
//   
//     ix = (171*ix) % 30269;
//     iy = (172*iy) % 30307;
//     iz = (170*iz) % 30323;
// 
//     r  = static_cast<double>(ix)/30269.
//          + static_cast<double>(iy)/30307. 
//          + static_cast<double>(iz)/30323.;
// 
//     return r - static_cast<int>(r);
// }

// initialize a Mersenne twister random number generator that returns
// an instance of mt19937. This function also prints out the random seed 
// sequence that will be used throughout
// seed_data is initialized here so that it can be put into the outfile in the
// main() section
std::vector<unsigned int> seed_data{};
std::mt19937& get_rng()
{
    static std::mt19937 generator;   // declared static
    static bool initialized = false; // guard to ensure one-time seeding

    if (!initialized)
    {
        std::random_device rd{};
        
        // fill the seed_data vector with 8 seeds (best practice)
        for (size_t i = 0; i < 8; i++)
        {
            seed_data.push_back(rd());
        }
        
        // generate a seed sequence using the 8 seeds generated above
        std::seed_seq seq(seed_data.begin(), seed_data.end());

        // set the seed for the Mersenne twister
        generator.seed(seq);

        // Print seed sequence once
        std::cout << "Seed sequence: ";
        for (size_t i = 0; i < seed_data.size(); i++)
        {
            std::cout << seed_data[i] << " ";
        }
        
        std::cout << "\n\n";

        initialized = true;
    }
    
    // return the generator to be used in the uniform(0, 1) random number 
    // generator below
    return generator;
}

// Uniform(0,1) generator
double uniform_random_01()
{
    static std::uniform_real_distribution<double> uniform(0.0, 1.0);
    return uniform(get_rng());
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

// pulled from: https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/#Computer_implementations
// error is 1.15e-9
// scipy uses a different approximation from W.J. Cody published in AMS
// "Rational Chebyshev approximations for the error function" by W. J. Cody
// https://www.ams.org/journals/mcom/1969-23-107/S0025-5718-1969-0247736-4/S0025-5718-1969-0247736-4.pdf
// maximal relative error is 6e-19 to 3e-20
double inverse_normal_cdf(double p)
{
    double  A1 {-3.969683028665376e+01};
    double  A2 {2.209460984245205e+02};
    double  A3 {-2.759285104469687e+02};
    double  A4 {1.383577518672690e+02};
    double  A5 {-3.066479806614716e+01};
    double  A6 {2.506628277459239e+00};

    double  B1 {-5.447609879822406e+01};
    double  B2 {1.615858368580409e+02};
    double  B3 {-1.556989798598866e+02};
    double  B4 {6.680131188771972e+01};
    double  B5 {-1.328068155288572e+01};

    double  C1 {-7.784894002430293e-03};
    double  C2 {-3.223964580411365e-01};
    double  C3 {-2.400758277161838e+00};
    double  C4 {-2.549732539343734e+00};
    double  C5 {4.374664141464968e+00};
    double  C6 {2.938163982698783e+00};

    double  D1 {7.784695709041462e-03};
    double  D2 {3.224671290700398e-01};
    double  D3 {2.445134137142996e+00};
    double  D4 {3.754408661907416e+00};

    double P_LOW {0.02425};
    double P_HIGH {0.97575};

    double x{};
    double q{};
    double r{};
    double u{};
    double e{};

    if (0 < p && p < P_LOW)
    {
        q = sqrt(-2*log(p));
        x = (((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6) / ((((D1*q+D2)*q+D3)*q+D4)*q+1);
    }
  
    else if (P_LOW <= p && p <= P_HIGH)
    {
        q = p - 0.5;
        r = q*q;
        x = (((((A1*r+A2)*r+A3)*r+A4)*r+A5)*r+A6)*q /(((((B1*r+B2)*r+B3)*r+B4)*r+B5)*r+1);
    }

    else if (P_HIGH < p && p < 1) 
    {
        q = sqrt(-2*log(1-p));
        x = -(((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6) / ((((D1*q+D2)*q+D3)*q+D4)*q+1);
    }

    // The relative error of the approximation has absolute value less than 
    // 1.15e−9.  One iteration of Halley’s rational method (third order) gives
    // full machine precision.
    // restricted to the range if 0 to 1 because the above correctly estimates
    // the values at 0 and 1 while the below "correction" fails.
    if (0 < p && p < 1) 
    {
        e = 0.5 * erfc(-x/sqrt(2)) - p;
        u = e * sqrt(2*M_PI) * exp(x*x/2);
        x = x - u/(1 + x*u/2);
    }
  
    return x;
}

// finds K-stage triangular design for given design parameters. The resulting 
// design is put in the vector parameters 
void find_triangular_design(
        double delta0,
        double delta1,
        double sigma,
        double K,
        double required_alpha,
        double required_beta,
        std::vector<double>& parameters)
{
    // used to calculate delta_tilde
    double inv_norm_cdf_alpha = inverse_normal_cdf(1 - required_alpha);
    double inv_norm_cdf_beta = inverse_normal_cdf(1 - required_beta);

    // correction for delta to achieve required alpha and beta (delta_tilde)
    // from Wason 2018 paper
    double delta_tilde = (2*inv_norm_cdf_alpha) / (inv_norm_cdf_alpha + inv_norm_cdf_beta); 
    
    double delta = delta_tilde * (delta1 - delta0);

    // calculating maximum information by splitting the terms
    double Imax_term1 = (4. * pow(0.583, 2.))/K;
    double Imax_term2 = 8. * log(1./(2. * required_alpha));
    double Imax_term3 = (2. * 0.583)/sqrt(K);

    double Imax = pow(sqrt(Imax_term1 + Imax_term2) - Imax_term3, 2.) / pow(delta, 2.);

    // finding the number of individuals per stage
    int num_indiv_per_stage = static_cast<int>(ceil(Imax*2*sigma*sigma/K));
    
    // create a vector for collecting the cumulative sample size 
    std::vector<double> cumu_sample_size;
    cumu_sample_size.push_back(num_indiv_per_stage);
   
    // fills a vector that contains number of individuals total, e.g.,
    // if num_indiv_per_stage = 20, and there are 3 stages, this generates a
    // vector called cumulative sample size = {20, 40, 60}
    // here i is size_t because we are indexing the vector below
    for(size_t i = 1; i < static_cast<size_t>(K); i++)
    {
        cumu_sample_size.push_back(
            num_indiv_per_stage + cumu_sample_size.at(i-1)
        );
    }

    parameters.clear();

    // the first parameter in vector is the number of individuals per stage
    // calculated above using information
    parameters.push_back(num_indiv_per_stage);

    double information{};
    double c{}; // lower bounds
    double d{}; // upper bounds

    // here i is double because it is being used to calculate information 
    // fraction
    for(double i = 0; i < K; i++)
    {
        // calculate the bounds on the score scale
        c=-(2.0/delta)*log(1.0/(2*required_alpha))+0.583*sqrt(Imax/K)+(3*delta/4)*((i+1)/K)*Imax;
        d=(2.0/delta)*log(1.0/(2*required_alpha))-0.583*sqrt(Imax/K)+(delta/4)*((i+1)/K)*Imax;
        
        // using the information at the current stage, convert the bounds to 
        // the standardized z scale
        information=Imax*((i+1)/K);
        parameters.push_back(c/sqrt(information));
        parameters.push_back(d/sqrt(information));
        
    }

}

// finds the sample size required for a one-stage trial with 
// given design parameters
double one_stage_sample_size(
        double difference,
        double sigma,
        double type_I_error,
        double type_II_error,
        double R)
{
    // ratio of smaller group to larger group
    double r = (1+R)/R;

    // z statistic for alpha
    double z_alpha = inverse_normal_cdf(1-type_I_error);

    // z statistic for power
    double z_power = inverse_normal_cdf(1-type_II_error);
    
    return r * ((pow(sigma, 2) * pow(z_alpha + z_power, 2))/(difference*difference));
}

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

    // separating indexing vector with multiplication without loop variable by 
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

void convert_h_to_psi(
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

void convert_h_to_phi(
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

// finds the delta which gives highest expected sample size for a given design
// this function was rewritten from the original and now uses interval 
// bisection
void find_delta_minimax_seq(
        std::vector<std::vector<double>>& h,
        std::vector<std::vector<double>>& z,
        std::vector<double>& parameters,
        double delta0,
        double delta1,
        double sigma,
        double *delta_minimax,
        double *max_expected_n)
{
    // this is the accuracy at which to find the delta 
    double epsilon {1e-4};
    
    // lower and upper bounds for the delta values
    // delta1 is multiplied by a large number to start at the edge
    double lower_delta = delta0;
    double upper_delta = delta1 * 10.; 
    
    // calculate the first sample sizes for the lower and upper bounds
    // lower bounds need a set of phi and psi and upper bounds need a set of
    // phi and psi
    std::vector<double> lower_phi;
    std::vector<double> upper_phi;
    
    std::vector<double> lower_psi;
    std::vector<double> upper_psi;
    
    double lower_ess = expected_sample_size(lower_phi, lower_psi, parameters);
    double upper_ess = expected_sample_size(upper_phi, upper_psi, parameters);
    
    // while the difference between the deltas being used to search is larger
    // than the accuracy, divide the interval into two and start again
    while( std::abs(lower_delta - upper_delta) > epsilon ) 
    {

        convert_h_to_phi(h, z, lower_phi, parameters, delta0, lower_delta, sigma);
        convert_h_to_psi(h, z, lower_psi, parameters, delta0, lower_delta, sigma);

        convert_h_to_phi(h, z, upper_phi, parameters, delta0, upper_delta, sigma);
        convert_h_to_psi(h, z, upper_psi, parameters, delta0, upper_delta, sigma);

        lower_ess = expected_sample_size(lower_phi, lower_psi, parameters);
        upper_ess = expected_sample_size(upper_phi, upper_psi, parameters);
        
        if (upper_ess > lower_ess)
        {
            lower_delta = (lower_delta + upper_delta) / 2;
        }

        else
        {
            upper_delta = (lower_delta + upper_delta) / 2;
        }
    }

    // these values are filled to be used in other functions
    *delta_minimax = upper_delta;
    *max_expected_n = upper_ess;
}

// trial_properties_seq uses the method given in Section 19.2 of Jennison and 
// Turnbull (2000) to find the probability of stopping at each stage in a 
// sequential trial using Z-tests
void trial_properties_seq(
        std::vector<double>& parameters,
        double delta0,
        double delta1,
        double sigma,
        double *type_I_error,
        double *power,
        double *expected_sample_size_null,
        double *expected_sample_size_crd,
        double *worse_case_delta,
        double *expected_sample_size_dm,
        int check_dm)
{

    // Function will find type_I_error and power for trial parameters. If 


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
    std::vector<std::vector<double>> z;
    
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
    std::vector<std::vector<double>> h;
    
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
    convert_h_to_phi(h, z, phi, parameters, delta0, delta0, sigma);
    
    // calculate type_I_error by summing up phi under the null
    *type_I_error = 0;
    
    for(i = 0; i < phi.size(); i++)
    {
        *type_I_error += phi.at(i);
    }
    
    convert_h_to_psi(h, z, psi, parameters, delta0, delta0, sigma);
    
    *expected_sample_size_null = expected_sample_size(phi, psi, parameters);
    
    convert_h_to_phi(h, z, phi, parameters, delta0, delta1, sigma);
    
    // calculate power by summing phi under the alternative
    *power = 0;
    
    for(i = 0; i < phi.size(); i++)
    {
        *power += phi.at(i);
    }

    convert_h_to_psi(h, z, psi, parameters, delta0, delta1, sigma);
    
    *expected_sample_size_crd = expected_sample_size(phi, psi, parameters);
    
    // if check_dm==1, the worst-case scenario delta will be found together with 
    // its expected sample size. Else, both will be returned as 0
    *worse_case_delta = 0;
    *expected_sample_size_dm = 0;
    double delta_minimax {};
    
    if(check_dm == 1)
    {
        find_delta_minimax_seq
        (
            h, z, parameters, delta0, delta1, sigma, 
            &delta_minimax, expected_sample_size_dm 
        );

        *worse_case_delta=delta_minimax;
    }

}

double function_value_delta_minimax(
        std::vector<double>& candidate_params,
        double delta0,
        double delta1,
        double sigma,
        double required_type_I_error,
        double required_type_II_error,
        double penalty_parameter,
        int numberrestarts)
{
    double type_I_error;
    double power;
    double expected_sample_size_null;
    double expected_sample_size_crd;
    double worse_case_delta;
    double expected_sample_size_dm;

    // get the trial properties for the candidate parameters using the 
    // function input
    trial_properties_seq(
        candidate_params,
        delta0,
        delta1,
        sigma,
        &type_I_error,
        &power,
        &expected_sample_size_null,
        &expected_sample_size_crd,
        &worse_case_delta,
        &expected_sample_size_dm,
        1 // when check_dm==1, finds worst case delta
    );
    
    // start the penalty at 0
    double functionvalue=0;

    if (type_I_error > required_type_I_error)
    {
        functionvalue += (penalty_parameter 
                          + (type_I_error-required_type_I_error)/required_type_I_error)
                          * penalty_parameter;
    }
    
    if ((1-power) > required_type_II_error)
    {
        functionvalue += penalty_parameter
                         + (((1-power)-required_type_II_error)/required_type_II_error)
                         * penalty_parameter;
    }
    
    if ((type_I_error > required_type_I_error || (1-power) > required_type_II_error) 
         && numberrestarts >= (-1))
    {
        functionvalue += penalty_parameter/10;
    }
    
    functionvalue += expected_sample_size_dm;
    
    return(functionvalue); 

}

// this function is used exclusively during the function that generates
// candidate designs
double check_design_constraints(std::vector<double>& design)
{
    double validdesign {1};
    size_t numberstages = (design.size()-1)/2;
    size_t i;
    
    // if the first lower bound is larger than or equal to the first upper 
    // bound, this is an invalid design
    if (design.at(1) >= design.at(2))
    {
        validdesign=0;
    }
    
    for (i = 1; i < numberstages - 1; i++)
    {
        // if each lower bound after the first is larger than or equal to the 
        // next upper bound, this is an invalid design 
        if (design.at(i*2+1) >= design.at(i*2+2))
        {
            validdesign = 0;
        }
        
        // these are odd values only (corresponding to the lower bounds)
        // each subsequent lower bound must be equal or greater than the prior
        // e.g., if the first lower bound is -2, then then next lower bound 
        // must be >= -2 or else the design will be invalid
        if (design.at(i*2+1) < design.at(i*2-1))
        {
            validdesign = 0;
        }
        
        // these are even values only (corresponding to the upper bounds)
        // each subsequent lower bound must be equal or greater than the prior
        // e.g., if the first upper bound is 2, then the next upper bound
        // must be <= 2 or else the design will be invalid
        if (design.at(i*2+2) > design.at(i*2))
        {
            validdesign = 0;
        }
    }
   
    // check if the last upper bound is smaller than the second to last upper
    // bound
    if (design.at(numberstages*2) > design.at((numberstages-1)*2))
    {
        validdesign=0;
    }
    
    return validdesign;

}

//  this is the Box-Muller transform for transforming two uniform(0, 1) random
// variables into a normal(0, 1) random variable
// this is considered more computationally efficient than inverse transform 
// sampling
double normal_01_rng()
{
    double u1{};
    double u2{};

    u1 = uniform_random_01();
    u2 = uniform_random_01();
    
    return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}


void gen_candidate_state_delta_minimax(
        std::vector<double>& current_params,
        std::vector<double>& candidate_params,
        std::vector<double>& lower_ranges,
        std::vector<double>& upper_ranges,
        std::vector<double>& param_sigmas,
        int fixsamplesize)
{
    // for each candidate generation, pick one stage, and perturb that stage's 
    // parameters and the sample size per stage

    size_t i{};
    double u{};
    double temp{};

    candidate_params = current_params;

    u = uniform_random_01();
    
    size_t numberofstages = (current_params.size() - 1)/2;
    
    double temp_stagetochange = floor(static_cast<double>(numberofstages) * u);
    
    size_t stagetochange = static_cast<size_t>(temp_stagetochange);
    
    if (stagetochange == (numberofstages - 1))
    {
        
        candidate_params = current_params;
        // perturb sample size and last stage threshold:
        
        if (fixsamplesize == 0)
        {
            i = 0;
            do
            {
                temp=normal_01_rng();
                temp=temp*param_sigmas.at(i);
                candidate_params.at(i)=temp+current_params.at(i);
            }
            while(candidate_params.at(i)>=upper_ranges.at(i) || candidate_params.at(i)<=lower_ranges.at(i));
        }
        
        i = stagetochange * 2 + 1;
        do
        {
            temp = normal_01_rng();
            temp = temp * param_sigmas.at(i);
            candidate_params.at(i) = temp + current_params.at(i);
            candidate_params.at(i+1) = candidate_params.at(i);
        }
        while((candidate_params.at(i) >= upper_ranges.at(i) || candidate_params.at(i) <= lower_ranges.at(i)) || check_design_constraints(candidate_params) == 0);
    }
    
    else
    {
        do
        {
            candidate_params = current_params;
            
            // perturb sample size and last stage threshold:
            if (fixsamplesize == 0)
            {
                i = 0;
                do
                {
                    temp=normal_01_rng();
                    temp=temp*param_sigmas.at(i);
                    candidate_params.at(i)=temp+current_params.at(i);
                }
                while(candidate_params.at(i)>=upper_ranges.at(i) || candidate_params.at(i)<=lower_ranges.at(i));
            }
            
            i = stagetochange * 2 + 1;
            do
            {
                temp = normal_01_rng();
                temp = temp * param_sigmas.at(i);
                candidate_params.at(i) = temp + current_params.at(i);
            }
            while(candidate_params.at(i) >= upper_ranges.at(i) || candidate_params.at(i) <= lower_ranges.at(i));
      
            i = stagetochange * 2 + 2;
            do
            {
                temp=normal_01_rng();
                temp=temp*param_sigmas.at(i);
                candidate_params.at(i)=temp+current_params.at(i);
            }
            while(candidate_params.at(i) >= upper_ranges.at(i) || candidate_params.at(i) <= lower_ranges.at(i));
        }
        while(check_design_constraints(candidate_params) == 0);
    }

}


void simulatedannealing_delta_minimax(
        double delta0,
        double delta1,
        double sigma,
        double required_type_I_error,
        double required_power,
        std::vector<double> &initial_parameters, // starts as 65 3.13269e-16 2.13029 1.12976 1.88293 1.84489 1.84489 
        std::vector<double> lower_ranges, // starts as 2 -4 -4 -4 -4 -4 -4
        std::vector<double> upper_ranges, // starts as 154.149 4 4 4 4 4 4
        std::vector<double>& initial_parameters_sigma, // starts as 30.8299 3 3 3 3 3 3
        double initial_cost_temp,
        double finalparametersigma,
        double final_cost_temp,
        int num_candidate_generations_per_restart,
        int minnumberrestarts,
        std::vector<double> &finalparameters,
        double *finalfunctionvalue,
        double penalty_parameter)
{
    size_t i;
    std::vector<double> current_params = initial_parameters;
    
    double new_func_value;

    double min_func_value = function_value_delta_minimax(
        initial_parameters,
        delta0,
        delta1,
        sigma,
        required_type_I_error,
        (1-required_power),
        penalty_parameter,
        -2 // last parameter is number of restarts
    );
        
    double x;
    double numbersincereduction=0;
    double current_func_value;
    double reductioninfunctionvalue;
    
    std::vector<double> param_sigmas;
    std::vector<double> min_params = current_params;
    std::vector<double> candidate_params;

    param_sigmas = initial_parameters_sigma;
    int numberrestarts {0};
    
    double previousrestart;
    double candidate_generations {0};
    double cost_temp = initial_cost_temp;
    double rhocost = pow(final_cost_temp/initial_cost_temp, 1.0/num_candidate_generations_per_restart);
    double rhosigma = pow(finalparametersigma/param_sigmas.at(0), 1.0/num_candidate_generations_per_restart);
    
    do
    {
        gen_candidate_state_delta_minimax(
            current_params,
            candidate_params,
            lower_ranges,
            upper_ranges,
            param_sigmas,
            0
        );
        
        new_func_value = function_value_delta_minimax(
            candidate_params,
            delta0,
            delta1,
            sigma,
            required_type_I_error,
            (1-required_power),
            penalty_parameter,
            numberrestarts-minnumberrestarts
        );
            
        for(i=0;i<param_sigmas.size();i++)
        {
            param_sigmas.at(i) *= rhosigma;
        }
        
        
        
        candidate_generations++;
        
        // move from the current state with the following probability
        // e^(f'(x) - f(x) / temp) if it is greater than a random uniform
        // variable. this is the crux of the simulated annealing step
        x = uniform_random_01();
        if (exp(-(new_func_value - current_func_value)/cost_temp) > x)
        {
            current_func_value = new_func_value;
            cost_temp *= rhocost;
            current_params = candidate_params;
            
            if (new_func_value < min_func_value)
            {
                min_params = current_params;
                min_func_value = new_func_value;
                numbersincereduction = 0;
            }
            
            else
            {
                numbersincereduction++;
            }
        }
        
        else
        {
            numbersincereduction++;
        }
        
        if (static_cast<int>(numbersincereduction) % 25 == 0)
        {
            current_params = min_params;
            current_func_value = min_func_value;
        }
        
        if(candidate_generations >= num_candidate_generations_per_restart)
        {
            //reset temperature
            current_params = min_params;
            current_func_value = min_func_value;
            cost_temp = initial_cost_temp;
            rhocost = pow(final_cost_temp/initial_cost_temp, 1.0/num_candidate_generations_per_restart);
            param_sigmas = initial_parameters_sigma;
            rhosigma = pow(finalparametersigma/param_sigmas.at(0), 1.0/num_candidate_generations_per_restart);
            
            candidate_generations = 0;
            numberrestarts++;
            
            std::cout << "Restart " << numberrestarts << ", function value = " << min_func_value << "\n";
            reductioninfunctionvalue = previousrestart - min_func_value;
            previousrestart = min_func_value;
        }
    }
    while(numberrestarts<=minnumberrestarts || reductioninfunctionvalue>0.005);
        
    min_params.at(0) = floor(min_params.at(0));
    min_func_value = function_value_delta_minimax(
        min_params,
        delta0,
        delta1,
        sigma,
        required_type_I_error,
        (1-required_power),
        penalty_parameter,
        1
    );

    current_func_value=min_func_value;
    current_params=min_params;
    
    // repeat, but fixing samplesize
    candidate_generations = 0;
    numberrestarts -= 4;
    
    do
    {
        gen_candidate_state_delta_minimax(
            current_params,
            candidate_params,
            lower_ranges,
            upper_ranges,
            param_sigmas,
            1
        );
            
        new_func_value = function_value_delta_minimax(
            candidate_params,
            delta0,
            delta1,
            sigma,
            required_type_I_error,
            (1-required_power),
            penalty_parameter,
            1
        );
        
        // cout<<min_func_value<<" "<<new_func_value<<"\n";
        for (i = 0; i < param_sigmas.size(); i++)
        {
            param_sigmas.at(i) *= rhosigma;
        }
        
        x = uniform_random_01();
        
        candidate_generations++;
        
        if (exp(-(new_func_value-current_func_value)/cost_temp) > x)
        {
            current_func_value = new_func_value;
            cost_temp *= rhocost;
            current_params = candidate_params;
            
            if (new_func_value < min_func_value)
            {
                min_params = current_params;
                min_func_value = new_func_value;
                numbersincereduction = 0;
            }
            
            else
            {
                numbersincereduction++;
            }
        }
        
        else
        {
            numbersincereduction++;
        }
        
        if (static_cast<int>(numbersincereduction) % 10 == 0)
        {
            current_params = min_params;
            current_func_value = min_func_value;
        }
        
        if (candidate_generations >= num_candidate_generations_per_restart)
        {
            //reset temperature
            current_params = min_params;
            current_func_value = min_func_value;
            cost_temp = initial_cost_temp;
            
            rhocost = pow(final_cost_temp/initial_cost_temp, 1.0/num_candidate_generations_per_restart);
            param_sigmas = initial_parameters_sigma;
            
            rhosigma = pow(finalparametersigma/param_sigmas.at(0), 1.0/num_candidate_generations_per_restart);
            candidate_generations = 0;
            numberrestarts++;
            
            std::cout << "Restart " << numberrestarts << ", function value = " << min_func_value << "\n";
            reductioninfunctionvalue = previousrestart - min_func_value;
            
            min_func_value = function_value_delta_minimax(
                min_params,
                delta0,
                delta1,
                sigma,
                required_type_I_error,
                (1-required_power),
                penalty_parameter,
                numberrestarts - minnumberrestarts
            );
                
            previousrestart = min_func_value;
        
        }

    } while (numberrestarts <= minnumberrestarts || reductioninfunctionvalue > 0);
  
    finalparameters = min_params;
    *finalfunctionvalue = min_func_value;

}

// this is the high-level sequence of events in main()
// 1. standardize to normal 0, 1
// 2. get the one stage sample size
// 3. find the triangular design parameters
// 4. get the trial properties of the triangular design
// 5. set the:
//    - inital sigma parameters
//    - lower ranges of the parameters (this is the box to search in, it seems)
//    - upper ranges of the parameters
//    - initial parameters (triangular design)
// 6. perform the simulated annealing
// 7. get the final trial properties
// 8. write the results to the file
int main(int argc, char *argv[])
{
    if(argc != 8)
    {
        std::cout<<"Usage: ./finddeltaminimaxdesign <delta0> <delta1> <sigma> <type_I_error> <power> <number of stages> <outfile>\n";
        return 0;
    }
    
    // pull all the values from the user input
    double delta0 = atof(argv[1]);
    double delta1 = atof(argv[2]);
    double initial_sigma = atof(argv[3]);
    double required_type_I_error = atof(argv[4]);
    double required_power = atof(argv[5]);
    int K = atoi(argv[6]);
    std::string outfilename = argv[7];
    
    // run the uniform random generator a randomly specified number of 
    // iterations in order to start at a different place each run. If this loop
    // was not here, the first value for the uniform_random_01 call would be 
    // exact same each run. Seeds have 10 values, so seed % 10000 will take the
    // last 4 digits and run this loop that may times befor starting the 
    // optimization
    // this is also called a "warm up"
    // the entire random number generation routine has been replaced the two 
    // functions above -- get_rng() and uniform_random_01().

    // this was the original code with a warm up (not necessary to warm up 
    // below as the seed sequence takes care of this)
    // for (i = 0; i < seed % 10000; i++)
    // {
    //     uniform_random_01();
    // }

    // std::cout << "Seed = " << seed << "\n";

    // standardise problem:
    double delta = (delta1-delta0)/initial_sigma;
    double sigma{1};

    double singlestagesamplesize = one_stage_sample_size(
        delta,
        sigma,
        required_type_I_error,
        (1-required_power),
        1
    );
    
    double type_I_error{};
    double power{};
    double expected_sample_size_null{};
    double expected_sample_size_crd{};
    double worse_case_delta{};
    double expected_sample_size_dm{};
    
    std::vector<double> parameters{};
    std::vector<double> current_params{};
    std::vector<double> candidate_params{};
    std::vector<double> lower_ranges{};
    std::vector<double> upper_ranges{};
    std::vector<double> param_sigmas{};
    std::vector<double> initial_param_sigmas{};
    
    find_triangular_design(
        0,
        delta,
        sigma,
        K,
        required_type_I_error * 49/50,
        (1-required_power),
        parameters
    );
    
    // find trial properties of triangular design
    trial_properties_seq(
        parameters,
        0,
        delta,
        sigma,
        &type_I_error,
        &power,
        &expected_sample_size_null,
        &expected_sample_size_crd,
        &worse_case_delta,
        &expected_sample_size_dm,
        1
    );
    
    // set lower ranges for parameters in simulated annealing
    lower_ranges.push_back(2);
    upper_ranges.push_back(singlestagesamplesize);
    
    initial_param_sigmas.push_back(singlestagesamplesize/5);
    for (int i = 0; i < K; i++)
    {
        lower_ranges.push_back(-4);
        upper_ranges.push_back(4);
        lower_ranges.push_back(-4);
        upper_ranges.push_back(4);
        
        initial_param_sigmas.push_back(3);
        initial_param_sigmas.push_back(3);
    }
    
    std::vector<double> initial_parameters = parameters;
    std::vector<double> finalparameters;
    
    double finalfunctionvalue;
    
    // DEBUG
    for (auto num : initial_parameters)
        std::cout << num << " ";

    std::cout << "\n";

    // DEBUG
    for (auto num : lower_ranges)
        std::cout << num << " ";

    std::cout << "\n";

    // DEBUG
    for (auto num : upper_ranges)
        std::cout << num << " ";

    std::cout << "\n";

    // DEBUG
    for (auto num : initial_param_sigmas)
        std::cout << num << " ";

    // DEBUG
    std::cout << "\n";

    // carries out simulated annealing to find sample size and stopping
    // boundaries for delta minimax design. First, the process allows n to be
    // non-integer, searching over the sample size and stopping boundaries.
    // After, the sample size is rounded to the nearest integer, and the 
    // stopping boundaries only are searched over.
    simulatedannealing_delta_minimax(
        0, // delta0
        delta,
        sigma,
        required_type_I_error,
        required_power,
        initial_parameters, // starts as 65 3.13269e-16 2.13029 1.12976 1.88293 1.84489 1.84489 
        lower_ranges,       // starts as 2 -4 -4 -4 -4 -4 -4
        upper_ranges,       // starts as 154.149 4 4 4 4 4 4
        initial_param_sigmas, // starts as 30.8299 3 3 3 3 3 3
        100, // initial cost temp
        0.005, // final parameters sigma 
        0.005, // final cost temp
        10000, // num candidate restarts
        5, // min num restarts
        finalparameters,
        &finalfunctionvalue,
        singlestagesamplesize
    );
    
    trial_properties_seq(
        finalparameters,
        0,
        delta,
        sigma,
        &type_I_error,
        &power,
        &expected_sample_size_null,
        &expected_sample_size_crd,
        &worse_case_delta,
        &expected_sample_size_dm,
        1
    );
    
    // write results to file with specified name
    std::ofstream outfile;
    outfile.open(outfilename.c_str(), std::ios_base::app);
    
    for (size_t i = 0; i < seed_data.size(); i++)
    {
        outfile << seed_data.at(i) << " ";
    }

    outfile << required_type_I_error << " " 
            << required_power << " " 
            << K << " "
            << type_I_error << " " 
            << power << " " 
            << expected_sample_size_null << " " 
            << expected_sample_size_crd << " " 
            << expected_sample_size_dm << " ";
            
    for(size_t i = 0; i < finalparameters.size(); i++)
    {
        outfile << finalparameters.at(i) << " ";
    }
    
    outfile << "\n";
    outfile.close();
    
    return 0;
}
