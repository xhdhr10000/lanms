#include <torch/script.h>

namespace lanms_adaptor {
    torch::Tensor merge_quadrangle_n9(torch::Tensor quad_n9, double iou_threshold);
};

int main(int argc, char** argv) {
    const double threshold = 0.3, precision = 10000;
    torch::Tensor input = torch::rand({10, 9}, at::kDouble);
    std::cout << "input:\n" << input << std::endl;
    input *= precision;
    torch::Tensor output = lanms_adaptor::merge_quadrangle_n9(input, threshold) / precision;
    std::cout << "output:\n" << output << std::endl;
    return 0;
}