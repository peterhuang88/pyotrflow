#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>

#include "Layer.h"
#include "Net.h"

int main(int argc, char** argv) {
    // Layer layer1(5);
    Net my_net(0.01);
    my_net.addLayer(5);
    my_net.addLayer(10);
    my_net.printNet();
    std::cout << "Test done\n";
    return 0;
}