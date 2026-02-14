#include <iostream>

void doNothing(int&) // Don't worry about what & is for now, we're just using it to trick the compiler into thinking variable x is used
{
}

int main () {
    [[maybe_unused]] int a;
    double b = 4.5;
    double c {4.5};

    // std::cout << a << std::endl;
    std::cout << b << "\n";
    std::cout << c << std::endl;

    // define an integer variable named x
    int x; // this variable is uninitialized because we haven't given it a value

    doNothing(x); // make the compiler think we're assigning a value to this variable

    // print the value of x to the screen
    std::cout << x << '\n'; // who knows what we'll get, because x is uninitialized

    return 0;
}
