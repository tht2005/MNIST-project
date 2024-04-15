#ifndef NEURAL_NETWORK_H_fjdlkfjsldfjlkicklk
#define NEURAL_NETWORK_H_fjdlkfjsldfjlkicklk

#include <algorithm>
#include <vector>

#include "myrandom.h"
#include "myassert.h"

class neural_network {
	private:
		// network graph
		size_t size, nlayer;
		std::vector<size_t> layer_sizes, layer_split; // indexs of node of layer i is [ layer_split[i], layer_split[i + 1] )

		// network sums, values, weights, biases
		std::vector<double> sums, values, biases;
		std::vector<std::vector<double>> weights;

		// partial derivative of loss function respect to biases
		std::vector<double> d_b;
		// partial derivative of loss function respect to weights
		std::vector<std::vector<double>> d_w;
		// partial derivative of loss function respect to values
		std::vector<double> dp;

		// network functions
		std::vector< double(*)(double) > layer_activate_functions;
		std::vector< double(*)(double) > layer_derivative_functions;
		double (*loss_function)(double, double);
		double (*loss_function_derivative)(double, double);

	public:
		neural_network(const std::vector<size_t>& layer_sizes,
				const std::vector< double(*)(double) >& layer_activate_functions,
				const std::vector< double(*)(double) >& layer_derivative_functions,
				double (*loss_function)(double, double),
				double (*loss_function_derivative)(double, double));

		void init_random_weight_bias();

		void get_input(const std::vector<double>& inputs);

		void calculate_values();
		void calculate_gradients(const std::vector<double>& desired_outputs);
		void gradient_descent(double ALPHA);

		double calculate_total_cost(const std::vector<double>& desired_outputs);

		size_t choose();
};

#endif

