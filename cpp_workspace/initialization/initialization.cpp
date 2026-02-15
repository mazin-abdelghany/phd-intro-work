#include <iostream>

// Don't worry about what & is for now, we're just using it to trick the 
// compiler into thinking variable x is used
void doNothing(int&) 
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
    // this variable is uninitialized because we haven't given it a value
    int x; 
    
    // make the compiler think we're assigning a value to this variable
    doNothing(x); 

    // print the value of x to the screen
    // who knows what we'll get, because x is uninitialized
    std::cout << x << '\n'; 

    return 0;
}
