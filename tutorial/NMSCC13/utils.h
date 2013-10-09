/*
 * utils.h
 *
 *  Created on: Oct 8, 2013
 *      Author: ollie
 */

#ifndef UTILS_H_
#define UTILS_H_

// functor to print integer with certain width
struct print_integer
{
    print_integer(int width) : w(width) {}

    void
    operator() (int x) {
	std::cout << std::setw(w) << x;
    }

private:
    int w;
};

// functor to print integer with certain width
struct print_float
{
    print_float(int width) : w(width) {}

    void
    operator() (float x) {
	std::cout << std::setw(w) << x;
    }

private:
    int w;
};

void pause()
{
    // pause for user to hit enter.
    std::cin.get();
}

#endif /* UTILS_H_ */
