#include <iostream>
#include <random> // for std::mt19937 and std::random_device
#include <vector>
#include <chrono>

double uniform_random_01()
{
    static long ix{1};
    static long iy{1};
    static long iz{1};

    double r{};
  
    ix = (171*ix) % 30269;
    iy = (172*iy) % 30307;
    iz = (170*iz) % 30323;

    r  = static_cast<double>(ix)/30269.
         + static_cast<double>(iy)/30307. 
         + static_cast<double>(iz)/30323.;

    return r - static_cast<int>(r);
}

std::vector<unsigned int> seed_data {};

// Returns reference to RNG seeded exactly once
std::mt19937& get_rng()
{
    static std::mt19937 generator;   // declared static
    static bool initialized = false; // guard to ensure one-time seeding

    if (!initialized)
    {
        std::random_device rd;
        
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
double uniform_0_1()
{
    static std::uniform_real_distribution<double> uniform(0.0, 1.0);
    return uniform(get_rng());
}

int main()
{
    std::vector<std::chrono::duration<double>> times{};
    for (int i = 0; i < 100; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        uniform_0_1();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        times.push_back(duration);
    }

    for (size_t i = 0; i < times.size(); i++)
    {
        std::cout << times[i].count() << " ";
    } 
}
