#include <vector>
#include <iostream>
#include <cmath>
#include <random>

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
        if (i == pair_checks - 1)
        {
            // the last two bounds must be equal
            if (design[2 * (i + 1)] != design[(2 * (i + 1)) - 1])
            {
                return 0;
            }

        }
    }

    return 1;    

}

void gen_candidate_state_delta_minimax(
        std::vector<double>& current_params, // starts as 65 3.13269e-16 2.13029 1.12976 1.88293 1.84489 1.84489 
        std::vector<double>& candidate_params, // starts as empty vector
        std::vector<double>& lower_ranges, // starts as 2 -4 -4 -4 -4 -4 -4
        std::vector<double>& upper_ranges, // starts as 154.149 4 4 4 4 4 4
        std::vector<double>& param_sigmas, // starts as 30.8299 3 3 3 3 3 3
        int fixsamplesize)
{
    // for each candidate generation, pick one stage, and perturb that stage's 
    // parameters and the sample size per stage

    size_t i{};
    double u{};
    double temp{};

    candidate_params = current_params;

    u = uniform_random_01();
    
    size_t numberofstages = ((current_params.size() - 1)/2);
    
    double temp_stagetochange = floor(static_cast<double>(numberofstages) * u);
    
    size_t stagetochange = static_cast<size_t>(temp_stagetochange);
    
    std::cout << "At the beginning, parameter values are:\n"
            << "u=" << u << "\n"
            << "num_stages=" << numberofstages << "\n"
            << "stage_to_change=" << stagetochange << "\n"
            << "candidate_params=";

    for (size_t k = 0; k < candidate_params.size(); k++)
    {
        std::cout << candidate_params[k] << " ";
    }
    std::cout << "\n\n";

    if (stagetochange == (numberofstages -1))
    {
        std::cout << "We are in the first if statement.\n";

        candidate_params = current_params;
        // perturb sample size and last stage threshold:
        
        std::cout << "candidate_params=";
        for (size_t k = 0; k < candidate_params.size(); k++)
        {
            std::cout << candidate_params[k] << " ";
        }
        std::cout << "\n";

        if (fixsamplesize == 0)
        {
            std::cout << "We are in the first nested if.\n";

            i = 0;
            do
            {
                std::cout << "We are changing the sample size.\n";
                temp=normal_01_rng();
                std::cout << "normal_rng=" << temp << "\n";
                temp=temp*param_sigmas.at(i);

                std::cout << "normal_rng * sigma=" << temp << "\n";
                candidate_params.at(i)=temp+current_params.at(i);
                std::cout << "new sample size=" << candidate_params[i] << "\n";
            }
            while(candidate_params.at(i)>=upper_ranges.at(i) || candidate_params.at(i)<=lower_ranges.at(i));
        }
        
        std::cout << "We changed the sample size\n";
        i = stagetochange * 2 + 1;
        std::cout << "i=" << i << "\n";

        std::cout << "New stage to change is " << i+1 << "\n";
        do
        {
            temp = normal_01_rng();
            std::cout << "We are changing stage " << i+1 << "\n";
            std::cout << "normal rng=" << temp << "\n";
            temp = temp * param_sigmas.at(i);
            std::cout << "normal rng * sigma=" << temp << "\n";
            candidate_params.at(i) = temp + current_params.at(i);
            std::cout << "candidate_params at " << i+1 << " " << candidate_params[i] << "\n";
            candidate_params.at(i+1) = candidate_params.at(i);
            std::cout << "candidate_params at " << i+2 << " " << candidate_params[i+1] << "\n";
        }
        while((candidate_params.at(i) >= upper_ranges.at(i) || candidate_params.at(i) <= lower_ranges.at(i)) || check_design_constraints(candidate_params) == 0);
    }
    
    else
    {
        std::cout << "We are in the else after the first if \n";
        std::cout << "We are changing stage " << stagetochange << "\n";
        do
        {
            candidate_params = current_params;
            
            std::cout << "candidate_params=";
            for (size_t k = 0; k < candidate_params.size(); k++)
            {
                std::cout << candidate_params[k] << " ";
            }
            std::cout << "\n";

            // perturb sample size and last stage threshold:
            if (fixsamplesize == 0)
            {
                std::cout << "We are in the nested if within the else\n";
                i = 0;
                do
                {  
                    temp=normal_01_rng();
                    std::cout << "normal rng " << temp << "\n";
                    temp=temp*param_sigmas.at(i);
                    std::cout << "normal rng * sigma" << temp << "\n";
                    candidate_params.at(i)=temp+current_params.at(i);
                    std::cout << "new sample size " << candidate_params[i] << "\n";
                }
                while(candidate_params.at(i)>=upper_ranges.at(i) || candidate_params.at(i)<=lower_ranges.at(i));
            }
            
            std::cout << "We changed the sample size\n";
            i = stagetochange * 2 + 1;
            std::cout << "i=" << i << "\n";
            do
            {
                temp = normal_01_rng();
                std::cout << "normal rng=" << temp << "\n";
                temp = temp * param_sigmas.at(i);
                std::cout << "normal rng * sigma" << temp << "\n";
                candidate_params.at(i) = temp + current_params.at(i);
                std::cout << "the new parameter is " << candidate_params.at(i) << "\n";
            }
            while(candidate_params.at(i) >= upper_ranges.at(i) || candidate_params.at(i) <= lower_ranges.at(i));
            
            std::cout << "We changed i=" << i << "\n";
            i = stagetochange * 2 + 2;
            std::cout << "Now we change i=" << i << "\n";
            do
            {
                temp=normal_01_rng();
                std::cout << "normal rng=" << temp << "\n";
                temp=temp*param_sigmas.at(i);
                std::cout << "normal rng * sigma" << temp << "\n";
                candidate_params.at(i)=temp+current_params.at(i);
                std::cout << "the new parameter is " << candidate_params.at(i) << "\n"; 
            }
            while(candidate_params.at(i) >= upper_ranges.at(i) || candidate_params.at(i) <= lower_ranges.at(i));

            std::cout << "We changed i=" << i << "\n";
            std::cout << "candidate_params=";
            for (size_t k = 0; k < candidate_params.size(); k++)
            {
                std::cout << candidate_params[k] << " ";
            }
            std::cout << "\n";
            
            
        }
        while(check_design_constraints(candidate_params) == 0);
    }

    std::cout << "Candidate params: ";
    for (size_t j = 0; j < candidate_params.size(); j++)
    {
        std::cout << candidate_params.at(j) << " ";
    }
    std::cout << "\n\n";

}

// for each candidate generation, pick one stage, and perturb that stage's 
// parameters and the sample size per stage
//
// param_sigmas is necessary because random perturbations start as normal(0, 1)
// and are transformed into necessary sizes with multiplication by corresponding
// sigma in the vector
void new_get_candidate_design(
        std::vector<double> current_params, // starts as 65 3.13269e-16 2.13029 1.12976 1.88293 1.84489 1.84489 
        std::vector<double> candidate_params, // starts as empty vector
        std::vector<double> lower_ranges, // starts as 2 -4 -4 -4 -4 -4 -4
        std::vector<double> upper_ranges, // starts as 154.149 4 4 4 4 4 4
        std::vector<double> param_sigmas, // starts as 30.8299 3 3 3 3 3 3
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

    for (size_t i = 0; i < candidate_params.size(); i++)
    {
        std::cout << candidate_params[i] << " ";
    }
    std::cout << "\n";
}


int main()
{

    std::vector<double> current_params {65, 3.13269e-16, 2.13029, 1.12976, 1.88293, 1.84489, 1.84489}; // starts as 65 3.13269e-16 2.13029 1.12976 1.88293 1.84489 1.84489 
    std::vector<double> candidate_params {}; // starts as empty vector
    std::vector<double> lower_ranges {2, -4, -4, -4, -4, -4, -4}; // starts as 2 -4 -4 -4 -4 -4 -4
    std::vector<double> upper_ranges {154.149, 4, 4, 4, 4, 4, 4}; // starts as 154.149 4 4 4 4 4 4
    std::vector<double> param_sigmas {30.8299, 3, 3, 3, 3, 3, 3}; // starts as 30.8299 3 3 3 3 3 3

    int how_many_loops {0};

    for (int i = 0; i < how_many_loops; i++)
    {
        new_get_candidate_design(
            current_params,
            candidate_params,
            lower_ranges,
            upper_ranges,
            param_sigmas,
            0
        );
    }

    new_get_candidate_design(
        current_params,
        candidate_params,
        lower_ranges,
        upper_ranges,
        param_sigmas,
        0
    );


    return 0;
}
