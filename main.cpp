#include <math.h>

#include "NeuralNetwork.h"
#include "myrandom.h"

// read 32-bit integer in MSB format
int read32(FILE *f) {
	int val;
	fread(&val, sizeof(int), 1, f);
	int ret = 0;
	while(val > 0) {
		ret = (ret << 8) | (val & 255);
		val >>= 8;
	}
	return ret;
}
void input(const char* label_name, const char* image_name, int& ndata, int label[], int& image_row, int& image_column, std::vector<double> pixels[]) {
	// read labels
	FILE *label_file = fopen(label_name, "rb");
	read32(label_file); // skip magic number
	ndata = read32(label_file);
	for(int i = 0; i < ndata; ++i) {
		fread(label + i, 1, 1, label_file);
	}
	fclose(label_file);

	// read images
	FILE *image_file = fopen(image_name, "rb");
	read32(image_file); // skip magic number
	myassert(ndata == read32(image_file),
			"Dev log: Why ndata are not equal in two files?");

	image_row = read32(image_file);
	image_column = read32(image_file);

	for(int i = 0; i < ndata; ++i) {
		pixels[i].clear();
		for(int x = 0; x < image_row; ++x) {
			for(int y = 0; y < image_column; ++y) {
				int a;
				fread(&a, 1, 1, image_file);
				pixels[i].push_back((double)a / 255);
			}
		}
	}

	fclose(image_file);
}

const int NDATA = 60006;
int ndata, ntest;
int label[NDATA], test_label[NDATA], image_row, image_column;
std::vector<double> pixels[NDATA], test_pixels[NDATA];

double ReLU(double x) {
	if(x > 0) {
		return x;
	}
	return 0.001 * x;
}
double d_ReLU(double x) {
	if(x > 0) {
		return 1;
	}
	return 0.001;
}

double sigmoid(double x) {
	double t = exp(-x);
	if(isinf(t)) {
		return 0;
	}
	return 1. / (1 + t);
}

double d_sigmoid(double x) {
	double t = exp(x);
	if(isinf(t)) {
		return 0;
	}
	return t / (1 + t) / (1 + t);
}

double loss_fun(double y, double y_label) {
	return (y - y_label) * (y - y_label);
}
double d_loss_fun(double y, double y_label) {
	return 2 * (y - y_label);
}

int main() {
	// read training data
	input("./data/train-labels.idx1-ubyte", "./data/train-images.idx3-ubyte", ndata, label, image_row, image_column, pixels);
	// read testing data
	input("./data/t10k-labels.idx1-ubyte", "./data/t10k-images.idx3-ubyte", ntest, test_label, image_row, image_column, test_pixels);

	std::vector<size_t> layer_sizes;
	// input layer
	layer_sizes.push_back(image_row * image_column);
	// hidden layers
	layer_sizes.push_back(350);
	layer_sizes.push_back(350);
	// output layer
	layer_sizes.push_back(10);	

	std::vector< double(*)(double) > layer_activate_functions;
	std::vector< double(*)(double) > layer_derivative_functions;

	layer_activate_functions.push_back(ReLU);
	layer_derivative_functions.push_back(d_ReLU);
	for(int i = 0; i < 2; ++i) {
		layer_activate_functions.push_back(sigmoid);
		layer_derivative_functions.push_back(d_sigmoid);
	}

	double (*loss_function)(double, double) = loss_fun;
	double (*loss_function_derivative)(double, double) = d_loss_fun;

	neural_network f(layer_sizes, layer_activate_functions,
			layer_derivative_functions, loss_function, loss_function_derivative);
	
	f.init_random_weight_bias();

	std::vector<double> outputs;

	std::cerr << std::endl;
	while(1) {
		double cost = 0;
		for(int i = 0; i < ndata; ++i) {
			f.get_input(pixels[i]);
			f.calculate_values();

			outputs.assign(10, 0);
			outputs[label[i]] = 1;

			f.calculate_gradients(outputs);
			f.gradient_descent(0.1);

			cost += f.calculate_total_cost(outputs);

			std::cerr << "\rImage: " << i;
		}

		std::cerr << std::endl << "Cost: " << cost << std::endl;

		int correct = 0;
		for(int i = 0; i < ntest; ++i) {
			f.get_input(test_pixels[i]);
			f.calculate_values();

			if(f.choose() == test_label[i]) {
				++correct;
			}
		}
		std::cerr << "Prediction: " << correct << " / " << ntest << std::endl;
	}
	return 0;
}
