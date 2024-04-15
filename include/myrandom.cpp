#include "myrandom.h"

#include <assert.h>
#include <chrono>
#include <random>

std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());

unsigned long long my_random() {
	return rng();
}

long long my_range_random(long long l, long long r) {
	assert(l <= r);
	return l + my_random() % (r - l + 1);
}

