#include "NeuralNetwork.h"

neural_network::neural_network(const std::vector<size_t>& layer_sizes,
		const std::vector< double(*)(double) >& layer_activate_functions,
		const std::vector< double(*)(double) >& layer_derivative_functions,
		double (*loss_function)(double, double),
		double (*loss_function_derivative)(double, double))

	: layer_sizes(layer_sizes),
	layer_activate_functions(layer_activate_functions),
	layer_derivative_functions(layer_derivative_functions),
	loss_function(loss_function),
	loss_function_derivative(loss_function_derivative)

{
	// initialize nlayer and assert something
	nlayer = layer_sizes.size();

	myassert(nlayer > 1,
			"Error: number of layers must be > 1");
	for(size_t i = 0; i < nlayer; ++i) {
		if(layer_sizes[i] == 0) {
			myassert(0,
					"Error: A layer's size must be greater than 0");
		}
	}
	myassert(nlayer == layer_activate_functions.size() + 1,
			"Error: layer_activate_functions.size() not equal to number of hidden layers");
	myassert(nlayer == layer_derivative_functions.size() + 1,
			"Error: layer_derivative_functions.size() not equal to number of hidden layers");

	// initialize size and layer_split
	size = 0;
	for(size_t i = 0; i < nlayer; ++i) {
		layer_split.push_back(size);
		size += layer_sizes[i];
	}
	layer_split.push_back(size);

	// resize
	sums.resize(size);
	values.resize(size);

	biases.resize(size);
	weights.resize(size);
	for(size_t layer = 1; layer < nlayer; ++layer) {
		for(size_t i = layer_split[layer]; i < layer_split[layer + 1]; ++i) {
			weights[i].resize(layer_sizes[layer - 1]);
		}	
	}
}

void neural_network::init_random_weight_bias() {
	const long long Radius = 1;
	const long long scale = 100000;

	for(std::vector<double>& v : weights) {
		for(double& w : v) {
			w = (double)my_range_random(-Radius * scale, Radius * scale) / scale;
		}
	}
	for(double& b : biases) {
		b = (double)my_range_random(-Radius * scale, Radius * scale) / scale;
	}
}

void neural_network::get_input(const std::vector<double>& inputs) {
	myassert(layer_sizes[0] == inputs.size(),
			"Error: In function neural_network::get_input, inputs.size() != [number of nodes in input layer]");

	// copy inputs to values[i] of input layer
	for(size_t i = 0; i < layer_sizes[0]; ++i) {
		values[i] = inputs[i];
	}
}

void neural_network::calculate_values() {
	for(size_t layer = 1; layer < nlayer; ++layer) {
		for(size_t i = layer_split[layer]; i < layer_split[layer + 1]; ++i) {
			myassert(weights[i].size() == layer_sizes[layer - 1],
					"Dev log: why weight[i].size() wrong?");

			// sum = sigma(weight * a) + bias
			sums[i] = biases[i];
			for(size_t j = 0; j < weights[i].size(); ++j) {
				// sum += weight * a
				sums[i] += weights[i][j] * values[ layer_split[layer - 1] + j ];
			}

			// a[i] = f(sum)
			values[i] = layer_activate_functions[layer - 1](sums[i]);
		}
	}
}

void neural_network::calculate_gradients(const std::vector<double>& desired_outputs) {
	myassert(desired_outputs.size() == layer_sizes.back(),
			"Desired vector's size must equal to output layer's size");
	myassert(nlayer + 1 == layer_split.size(),
			"Dev log: nlayer + 1 != layer_split.size()");

	dp.assign(size, 0);
	for(size_t i = layer_split[nlayer - 1]; i < layer_split.back(); ++i) {
		dp[i] = loss_function_derivative(values[i], desired_outputs[i - layer_split[nlayer - 1] ]);
	}

	myassert(nlayer > 0,
			"Dev log: Why nlayer == 0?");

	for(size_t layer = nlayer; layer --> 1; ) {
		for(size_t i = layer_split[layer]; i < layer_split[layer + 1]; ++i) {
			myassert(weights[i].size() == layer_sizes[layer - 1],
					"Dev log: why weight[i].size() wrong?");

			for(size_t j = 0, k; j < weights[i].size(); ++j) {
				k = layer_split[layer - 1] + j;
				dp[k] += dp[i] * weights[i][j] * layer_derivative_functions[layer - 1](sums[i]);
			}
		}
	}

	// resize d_b, d_w
	d_b.resize(size);
	d_w.resize(weights.size());
	for(size_t i = weights.size(); i--; ) {
		d_w[i].resize(weights[i].size());
	}

	myassert(nlayer > 0,
			"Dev log: Why nlayer == 0?");
	// calculate d_w, d_b
	for(size_t i = layer_split[nlayer - 1]; i < layer_split.back(); ++i) {
		d_b[i] = loss_function_derivative(values[i], desired_outputs[ i - layer_split[nlayer - 1] ])
			* layer_derivative_functions.back()(sums[i]);
	}
	for(size_t layer = nlayer; layer --> 1; ) {
		for(size_t i = layer_split[layer]; i < layer_split[layer + 1]; ++i) {
			myassert(weights[i].size() == layer_sizes[layer - 1],
					"Dev log: why weight[i].size() wrong?");

			d_b[i] = dp[i] * layer_derivative_functions[layer - 1](sums[i]);

			for(size_t j = 0, k; j < weights[i].size(); ++j) {
				k = layer_split[layer - 1] + j;
				d_w[i][j] = dp[i] * values[k] * layer_derivative_functions[layer - 1](sums[i]);
			}
		}
	}
}

double neural_network::calculate_total_cost(const std::vector<double>& desired_outputs) {
	myassert(desired_outputs.size() == layer_sizes.back(),
			"Desired vector's size must equal to output layer's size");

	double ret = 0;
	for(size_t i = layer_split[nlayer - 1]; i < size; ++i) {
		ret += loss_function(values[i], desired_outputs[ i - layer_split[nlayer - 1] ]);
	}
	return ret;
}

void neural_network::gradient_descent(double ALPHA) {
	// update weights and biases based on gradient
	for(size_t layer = nlayer; layer --> 1; ) {
		for(size_t i = layer_split[layer]; i < layer_split[layer + 1]; ++i) {
			myassert(weights[i].size() == layer_sizes[layer - 1],
					"Dev log: why weight[i].size() wrong?");

			biases[i] -= ALPHA * d_b[i];
			for(size_t j = 0; j < weights[i].size(); ++j) {
				weights[i][j] -= ALPHA * d_w[i][j];
			}
		}
	}
}

size_t neural_network::choose() {
	auto it = max_element(values.begin() + layer_split[nlayer - 1], values.end());
	return it - values.begin() - layer_split[nlayer - 1];
}

