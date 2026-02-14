#include <iostream>

int main() {
    std::cout << "Enter a number: ";

    int x{};
    std::cin >> x;

    std::cout << "You entered: " << x << "\n";

    std::cout << "Now enter 2 numbers: ";

    int a {}, b {};

    std::cin >> a >> b;

    std::cout << "You entered " << a << " and " << b << std::endl;

    return 0;
}
