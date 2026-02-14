#include <iostream>

int main () {
    [[maybe_unused]] int a;
    double b = 4.5;
    double c {4.5};

    // std::cout << a << std::endl;
    std::cout << b << "\n";
    std::cout << c << std::endl;

    return 0;
}
