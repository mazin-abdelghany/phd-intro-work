#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>

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
        int n_restarts)
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

    //std::cout << "type I error=" << type_I_error << "\n";
    //std::cout << "power=" << power << "\n";
    //std::cout << "penalty_parameter=" << penalty_parameter << "\n";
    //std::cout << "function value=" << functionvalue << "\n";

    if (type_I_error > required_type_I_error)
    {
        functionvalue += (penalty_parameter 
                          + (type_I_error-required_type_I_error)/required_type_I_error)
                          * penalty_parameter;
    }

    //std::cout << "after if 1, function value=" << functionvalue << "\n";
    
    if ((1-power) > required_type_II_error)
    {
        functionvalue += penalty_parameter
                         + (((1-power)-required_type_II_error)/required_type_II_error)
                         * penalty_parameter;
    }

    //std::cout << "after if 2, function value=" << functionvalue << "\n";
    
    if ((type_I_error > required_type_I_error || (1-power) > required_type_II_error) 
         && n_restarts >= (-1))
    {
        functionvalue += penalty_parameter/10;
    }

    //std::cout << "after if 3, function value=" << functionvalue << "\n";
    
    functionvalue += expected_sample_size_dm;

    //std::cout << "expected sample size=" << expected_sample_size_dm << "\n";
    //std::cout << "after adding expected sample size, function value=" << functionvalue << "\n";
    
    return(functionvalue); 

}

// this function is used exclusively during the function that generates
// candidate designs
double check_design_constraints(std::vector<double>& design)
{
    // we check pairs, so only need to loop half of the vector length
    // the vector also contains the sample size at index 0, so 1 must be
    // subtracted
    size_t pair_checks = (design.size() - 1) / 2;

    // we loop through all of the boundaries, starting with i=1 because the 
    // first index is the sample size.
    for (size_t i = 1; i < pair_checks; i++)
    {

        // all of the below if statements check for invalid designs. as soon
        // as an invalid design is reached, then the function can stop checking
        // and therefore returns. this is more efficienct than checking every
        // constraint

        // all lower bounds must be less than or equal to their successor. the
        // lower bounds are at odd indexes. to perform the check: for example
        // ell_1 <= ell_2 is valid, thus if ell_1 > ell_2 (index 1 > index 3),
        // this design is invalid.
        if (design[2 * i - 1] > design[2*(i + 1) - 1])
        {
            return 0;
        }

        // all upper bounds must be greater than or equal to their predecessor.
        // the upper bounds are at even indexes. to perform the check: for 
        // example u_1 >= u_2 is valid, thus if u_1 < u_2 (index 2 < index 4),
        // this design is invalid
        if (design[2 * i] < design[2 * (i + 1)])
        {
            return 0;
        }

        // all upper bounds must be greater than all lower bounds unless we are
        // at the last bound 
        if (design[2 * i] <= design[2 * i - 1])
        {
            return 0;
        }
        
        // if we are at the last iteration
        //if (i == pair_checks - 1)
        //{
        //    // the last two bounds must be equal
        //    if (design[2 * (i + 1)] != design[(2 * (i + 1)) - 1])
        //    {
        //        return 0;
        //    }
        //
        //}

    }

    return 1;    

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
    // select a random stage to modify by generating a uniform(0, 1) random
    // variable and multiplying the number of stages by it, then take its
    // floor
    double u{};
    u = uniform_random_01();

    // number of stages requires -1 because the sample size is in the vector
    // divide by two because there are two boudaries per stage
    size_t num_stages = (current_params.size() - 1)/2;

    // temp variable is created to respect typing of .size() and floor()
    double temp_stagetochange = floor(static_cast<double>(num_stages) * u);
    size_t stagetochange = static_cast<size_t>(temp_stagetochange);
    
    // Fill candidate parameters with all zeros to pass the first while 
    // loop check. We set the candidate_params to the current_params to bypass
    // needing to know the size of the vector 
    candidate_params = current_params;
    for (size_t i = 0; i < candidate_params.size(); i++)
    {
        candidate_params.at(i) = 0;
    }

    // start generating candidate design
    // run the loop while the design is not valid, exit the loop once a valid
    // design has been generated
    while (check_design_constraints(candidate_params) == 0)
    {
        double temp{};
        size_t i{};

        // reset the candidates to the current (initially this is the 
        // triangular design)
        candidate_params = current_params;

        // check if we are modifying the last stage. If so, then the pair of
        // bounds will be the same
        if (stagetochange == num_stages - 1)
        {

            if (fixsamplesize == 0)
            {

                // do-while is used here because we need to perturb at least 
                // once prior to check the conditions
                do
                {
                    // generate a normal(0, 1) random variable, multiply it by
                    // the sigma and then add this random noise to the bound
                    temp = normal_01_rng();
                    temp = temp * param_sigmas.at(0);
                    candidate_params.at(0) = temp + current_params.at(0);
                }
                // make sure we remain within the box defined by upper_ and
                // lower_ranges
                while(candidate_params.at(0)>=upper_ranges.at(0) || candidate_params.at(0)<=lower_ranges.at(0));
                
            }
            
            i = stagetochange * 2 + 1;

            // same pattern as above
            do
            {
                temp = normal_01_rng();
                temp = temp * param_sigmas.at(i);
                candidate_params.at(i) = temp + current_params.at(i);
                candidate_params.at(i+1) = candidate_params.at(i);
            }
            while (candidate_params.at(i) >= upper_ranges.at(i) || candidate_params.at(i) <= lower_ranges.at(i));
            
        }

        // if we are not changing the last stage, start here.
        else
        {
            if (fixsamplesize == 0)
            {
       
                // same pattern as above
                do
                {
                    temp = normal_01_rng();
                    temp = temp * param_sigmas.at(0);
                    candidate_params.at(0) = temp+current_params.at(0);
                }
                while(candidate_params.at(0)>=upper_ranges.at(0) || candidate_params.at(0)<=lower_ranges.at(0));
                
            }
           
            // change the lower bound
            i = stagetochange * 2 + 1;

            do
            {
                temp = normal_01_rng();
                temp = temp * param_sigmas.at(i);
                candidate_params.at(i) = temp + current_params.at(i);
            }
            while(candidate_params.at(i) >= upper_ranges.at(i) || candidate_params.at(i) <= lower_ranges.at(i));
            
            // change the upper bound
            i = stagetochange * 2 + 2;
            
            do
            {
                temp = normal_01_rng();
                temp = temp * param_sigmas.at(i);
                candidate_params.at(i) = temp+current_params.at(i);
            }
            while(candidate_params.at(i) >= upper_ranges.at(i) || candidate_params.at(i) <= lower_ranges.at(i));
            
        }

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
        int min_n_restarts,
        std::vector<double> &finalparameters,
        double &finalfunctionvalue,
        double penalty_parameter)
{
    /////////////
    // VECTORS //
    // parameter manipulations
    std::vector<double> current_params = initial_parameters;
    std::vector<double> min_params = current_params;

    // search space will be reduced by reducing the param_sigma vector with
    // each design generation loop
    std::vector<double> param_sigmas;
    param_sigmas = initial_parameters_sigma;
    
    std::vector<double> candidate_params {};
    // END VECTORS //
    ////////////////
    
    //////////////
    // COUNTERS //
    // counts for the number of restarts and the number of generated study
    // design candidates. The maximum number of assessments that can occur,
    // crudely, is n_restarts * n_generated_candidates
    int n_restarts {0};
    double n_generated_candidates {0};
    double n_loops_since_last_func_improvement {0};
    int reduction_count {0};
    // END COUNTERS //
    //////////////////

    /////////////////////////
    // SIMULATED ANNEALING //
    // simulated annealing and search space parameters
    double cost_temp = initial_cost_temp;
    double rhocost = pow(final_cost_temp/initial_cost_temp, 1.0/num_candidate_generations_per_restart);
    double rhosigma = pow(finalparametersigma/param_sigmas.at(0), 1.0/num_candidate_generations_per_restart);
    // END SIMULATED ANNEALING //
    /////////////////////////////

    //////////////////////////////////////   
    // OBJECTIVE FUNCTION VALUE HOLDERS //
    // calculate the first minimum functinon value
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

    // holds the new objective function value after a new design candidate is
    // generated
    double new_func_value {};

    // hold the current working function value (i.e., the objective function
    // value from the last loop)
    double current_func_value {};

    // at each restart (i.e., after n_candidate_generations), the total 
    // magnitude by which the function value has been reduced is saved using
    // these two variables
    double reduction_in_func_value {};
    double min_func_value_at_last_restart {};
    // END OBJECTIVE FUNCTION VALUE HOLDERS //
    //////////////////////////////////////////

    ///////////////////
    // save vectors for path to minimum assessment
    std::vector<std::vector<double>> candidate_designs {};
    std::vector<double> objective_function_values {};
    std::vector<double> cost_temp_vec {};
    //////////////////

    // to hold the uniform(0, 1) random variable as part of the probabilistic
    // assessment for the simulated annealing move
    double x {};

    std::cout << "######################\n"
              << "First loop starting...\n"
              << "######################\n\n";
    
    while (n_restarts <= min_n_restarts || reduction_in_func_value < -0.005)
    {

        
        gen_candidate_state_delta_minimax(
            current_params,
            candidate_params,
            lower_ranges,
            upper_ranges,
            param_sigmas,
            0 // allow for sample size to change
        );
        
        new_func_value = function_value_delta_minimax(
            candidate_params,
            delta0,
            delta1,
            sigma,
            required_type_I_error,
            (1-required_power),
            penalty_parameter,
            n_restarts - min_n_restarts
        );

        if (n_generated_candidates < 500)
        {
            objective_function_values.push_back(new_func_value);
            candidate_designs.push_back(candidate_params);
            cost_temp_vec.push_back(cost_temp);
        }
        else if (static_cast<int>(n_generated_candidates) % 100 == 0)
        {
            objective_function_values.push_back(new_func_value);
            candidate_designs.push_back(candidate_params);
            cost_temp_vec.push_back(cost_temp);
        }
            
        for (size_t i = 0; i < param_sigmas.size(); i++)
        {
            param_sigmas.at(i) *= rhosigma;
        }
        
        n_generated_candidates++;
        
        // move from the current state with the following probability
        // e^(f'(x) - f(x) / temp) if it is greater than a random uniform
        // variable. this is the crux of the simulated annealing step
        x = uniform_random_01();
        if (exp(-(new_func_value - current_func_value)/cost_temp) > x)
        {
            current_func_value = new_func_value;
            cost_temp *= rhocost; // reduce the temperature
            current_params = candidate_params;
            
            // save the design if it is a new minimum
            if (new_func_value < min_func_value)
            {
                min_params = current_params;
                min_func_value = new_func_value;
                n_loops_since_last_func_improvement = 0;
            }
            
            else
            {
                n_loops_since_last_func_improvement++;
            }
        }
        
        else
        {
            n_loops_since_last_func_improvement++;
        }
        
        // if the design has not been reduced in the last 25 attempts, reset
        // the current parameters back to the minimum and try again from there
        if (static_cast<int>(n_loops_since_last_func_improvement) % 25 == 0)
        {
            current_params = min_params;
            current_func_value = min_func_value;
        }
        
        // if num_candidate_generations_per_restart is 1000, for example,
        // n_restarts is incremented every 1000 candidate generations. 
        // Therefore, unless the function reduction is < 0.005, the loop will
        // run for (n_restarts * num_candidate_generations_per_restart) times
        if (n_generated_candidates >= num_candidate_generations_per_restart)
        {
            // set the current parameters to the minimum found so far
            current_params = min_params;
            current_func_value = min_func_value;
            
            // reset the temperature and search space parameters back to 
            // initial values
            cost_temp = initial_cost_temp;
            param_sigmas = initial_parameters_sigma;
            
            // reset the number of generated study design candidates to 0
            n_generated_candidates = 0;

            n_restarts++;
            std::cout << "Restart " << n_restarts 
                      << ", current minimum objective function value = " 
                      << min_func_value << "\n";
            
            // calculate the decrease in the objective function value
            reduction_in_func_value = min_func_value - min_func_value_at_last_restart;
            std::cout << "Current minumum minus last minimum=" << reduction_in_func_value << "\n";
            std::cout << "Negative value means reduction in objective function\n\n";

            // reset the restart minimum function value
            min_func_value_at_last_restart = min_func_value;
        }
    } 
        
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
    n_generated_candidates = 0;
    n_restarts -= 4;

    std::cout << "######################\n"
              << "Second loop starting...\n"
              << "######################\n\n";

    std::cout << "Searching with a fixed, integer sample size...\n";
    
    while (n_restarts <= min_n_restarts || reduction_in_func_value < 0 || reduction_count < 2)
    {
        gen_candidate_state_delta_minimax(
            current_params,
            candidate_params,
            lower_ranges,
            upper_ranges,
            param_sigmas,
            1 // fix sample size
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
        
        for (size_t i = 0; i < param_sigmas.size(); i++)
        {
            param_sigmas.at(i) *= rhosigma;
        }
        
        n_generated_candidates++;
        
        // move from the current state with the following probability
        // e^(f'(x) - f(x) / temp) if it is greater than a random uniform
        // variable. this is the crux of the simulated annealing step 
        x = uniform_random_01();
        if (exp(-(new_func_value - current_func_value)/cost_temp) > x)
        {
            current_func_value = new_func_value;
            cost_temp *= rhocost; // reduce the temperature
            current_params = candidate_params;
            
            // save the design if it is a new minimum
            if (new_func_value < min_func_value)
            {
                min_params = current_params;
                min_func_value = new_func_value;
                n_loops_since_last_func_improvement = 0;
            }
            
            else
            {
                n_loops_since_last_func_improvement++;
            }
        }
        
        else
        {
            n_loops_since_last_func_improvement++;
        }
        
        // !!! above this is 25 attempts, now it is 10 attempts !!!
        // if the design has not been reduced in the last 10 attempts, reset
        // the current parameters back to the minimum and try again from there
        if (static_cast<int>(n_loops_since_last_func_improvement) % 10 == 0)
        {
            current_params = min_params;
            current_func_value = min_func_value;
        }
        
        if (n_generated_candidates >= num_candidate_generations_per_restart)
        {
            // set the current parameters to the minimum found so far
            current_params = min_params;
            current_func_value = min_func_value;
            
            // reset the temperature and search space parameters back to 
            // initial values
            cost_temp = initial_cost_temp;
            param_sigmas = initial_parameters_sigma;
            
            // reset the number of generated study design candidates to 0
            n_generated_candidates = 0;
            
            n_restarts++;
            std::cout << "Restart " << n_restarts 
                      << ", current minimum objective function value = " 
                      << min_func_value << "\n";
            
            // calculate the decrease in the objective function value
            reduction_in_func_value = min_func_value - min_func_value_at_last_restart;
            std::cout << "Current minumum minus last minimum=" << reduction_in_func_value << "\n";
            std::cout << "Negative value means reduction in objective function\n\n";

            // ensure that there are two iterations in a row where the objective
            // function is not reduced in value at all
            if (reduction_in_func_value == 0)
            {
                reduction_count++;
            }

            else
            {
                reduction_count = 0;
            }

            // reset the restart minimum function value
            min_func_value_at_last_restart = min_func_value;
            
        }

    }
  
    finalparameters = min_params;
    finalfunctionvalue = min_func_value;

    std::cout << "\n###############\n";
    std::cout << "PRINTING CANDIDATE DESIGNS\n\n";
    std::ofstream file1("designs.txt");
    for (size_t i = 0; i < candidate_designs.size(); i++)
    {
        for (size_t j = 0; j < candidate_designs[0].size(); j++)
        {
            file1 << candidate_designs[i][j] << " ";
        }
        file1 << "\n";
    }
    file1.close();

    std::cout << "\n##################\n";
    std::cout << "PRINTING OBJECTIVE FUCTION VALUES\n\n";
    std::ofstream file2("objective_function_vals.txt");
    for (size_t i = 0; i < objective_function_values.size(); i++)
    {
        file2 << objective_function_values[i] << "\n";
    }
    file2.close();

    std::cout << "\n################\n";
    std::cout << "PRINTING TEMPERATURE";
    std::ofstream file3("temperature.txt");
    for (size_t i = 0; i < cost_temp_vec.size(); i++)
    {
        file3 << cost_temp_vec[i] << "\n";
    }
    file3.close();
}

int main()
{

    double delta0 {0};
    double delta1 {1};
    double sigma {3};
    double required_type_I_error {0.05};
    double required_type_II_error {0.1};

    std::vector<double> initial_params {65, 3.13269e-16, 2.13029, 1.12976, 1.88293, 1.84489, 1.84489};
    std::vector<double> lower_ranges {2, -4, -4, -4, -4, -4, -4};
    std::vector<double> upper_ranges {154.149, 4, 4, 4, 4, 4, 4};
    std::vector<double> initial_parameters_sigma {30.8299, 3, 3, 3, 3, 3, 3};

    double initial_cost_temp {100};
    double final_parameters_sigma {0.005};
    double final_cost_temp {0.005};
    int num_cand_generations_per_restart {10000};
    int min_num_restarts {5};
    std::vector<double> final_parameters {};
    double final_function_val {};
    double penalty_parameter {154};


    simulatedannealing_delta_minimax(
        delta0,
        delta1,
        sigma,
        required_type_I_error,
        1-required_type_II_error,
        initial_params,
        lower_ranges,
        upper_ranges,
        initial_parameters_sigma,
        initial_cost_temp,
        final_parameters_sigma,
        final_cost_temp,
        num_cand_generations_per_restart,
        min_num_restarts,
        final_parameters,
        final_function_val,
        penalty_parameter
    );

    for (size_t i = 0; i < final_parameters.size(); i++)
    {
        std::cout << final_parameters[i] << " ";
    }
    std::cout << "\n";

    return 0;
}
