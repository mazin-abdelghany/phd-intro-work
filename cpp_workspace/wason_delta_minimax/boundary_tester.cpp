#include <vector>
#include <iostream>

double check_design_constraints(std::vector<double>& design)
{
    double validdesign {1};
    size_t numberstages = (design.size()-1)/2;
    size_t i;
    
    std::cout << "checking if l1="
            << design.at(1)
            << " is greater than or equal to u1=" 
            << design.at(2) << "\n";

    // if the first lower bound is larger than or equal to the first upper 
    // bound, this is an invalid design
    if (design.at(1) >= design.at(2))
    {
        validdesign=0;
    }
    
    for (i = 1; i < numberstages-1; i++)
    {
        std::cout << "checking if index " << i*2+1 << "="
                  << design.at(i*2+1)
                  << " is greater than or equal to index " << i*2 << "="
                  << design.at(i*2+2) << "\n";

        // if each lower bound after the first is larger than or equal to the 
        // next upper bound, this is an invalid design 
        if (design.at(i*2+1) >= design.at(i*2+2))
        {
            validdesign = 0;
        }

        std::cout << "checking if index " << i*2+1 << "="
            << design.at(i*2+1)
            << " is less than index " << i*2-1 << "="
            << design.at(i*2-1) << "\n";

        // these are odd values only (corresponding to the lower bounds)
        // each subsequent lower bound must be equal or greater than the prior
        // e.g., if the first lower bound is -2, then then next lower bound 
        // must be >= -2 or else the design will be invalid
        if (design.at(i*2+1) < design.at(i*2-1))
        {
            validdesign = 0;
        }

        std::cout << "checking if index " << i*2 << "="
        << design.at(i*2+2)
        << " is greater than index " << i <<  "="
        << design.at(i*2) << "\n";

        // these are even values only (corresponding to the upper bounds)
        // each subsequent lower bound must be equal or greater than the prior
        // e.g., if the first upper bound is 2, then the next upper bound
        // must be <= 2 or else the design will be invalid
        if (design.at(i*2+2) > design.at(i*2))
        {
            validdesign = 0;
        }
    }
  
    std::cout << "checking if index " << numberstages << "="
        << design.at(numberstages*2)
        << " is greater than index " << numberstages-1 << "="
        << design.at((numberstages-1)*2) << "\n";

    // check if the last upper bound is smaller than the second to last upper
    // bound
    if ( design.at(numberstages*2) > design.at((numberstages-1)*2) )
    {
        validdesign=0;
    }
    
    return validdesign;

}

int checkBounds(const std::vector<double>& bounds) {

    // we check pairs, so only need to loop half of the vector length
    // the vector also contains the sample size at index 0, so 1 must be
    // subtracted
    size_t pair_checks = (bounds.size() - 1) / 2;

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
        if (bounds[2 * i - 1] > bounds[2*(i + 1) - 1])
        {
            return 0;
        }

        // all upper bounds must be greater than or equal to their predecessor.
        // the upper bounds are at even indexes. to perform the check: for 
        // example u_1 >= u_2 is valid, thus if u_1 < u_2 (index 2 < index 4),
        // this design is invalid
        if (bounds[2 * i] < bounds[2 * (i + 1)])
        {
            return 0;
        }

        // all upper bounds must be greater than all lower bounds unless we are
        // at the last bound 
        if (bounds[2 * i] <= bounds[2 * i - 1])
        {
            return 0;
        }

        // the last two bounds must be equal
        if (bounds[2 * (i + 1)] != bounds[(2 * (i + 1)) - 1])
        {
            return 0;
        }
    }

    return 1;    

}

int main()
{
                                            //  l1           u1        l2       u2       l3        u3
    std::vector<double> parameters = { 61.1935, 3.13269e-16, 2.13029, 1.12976, 1.88293, -2.37106, -2.37106  };
    std::cout << check_design_constraints(parameters);
    std::cout << "\n\n";
    std::cout << checkBounds(parameters);
    std::cout << "\n\n";
    return 0;
}
