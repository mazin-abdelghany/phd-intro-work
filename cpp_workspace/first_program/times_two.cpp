#include <iostream>

int times_two(int x) 
{
    return x * 2;
}

int times_three (int x)
{
    return x * 3;
}

int main() 
{
    std::cout << "Enter an integer: ";
    int num { };
    std::cin >> num;
    
    std::cout << "Enter another integer: ";
    int num_2 { };
    std::cin >> num_2;


    std::cout << "Double " << num << " is: " << times_two(num) << "\n";
    
    std::cout << "Triple " << num << " is: " << times_three(num) << "\n";
    
    std::cout << num << " + " << num_2 << " is " << num + num_2 << ".\n";
    std::cout << num << " - " << num_2 << " is " << num - num_2 << ".\n";
    return 0;
}
