#include <matplot/matplot.h>
namespace plt = matplot;

int main(int argc, char **argv)
{
    try
    {
        std::vector<double> x = plt::linspace(0, 2 * plt::pi);
        std::vector<double> y = plt::transform(x, [](auto x) { return sin(x); });

        auto f = plt::figure(true);
        // plt::plot(x, y, "-o");
        // plt::hold(plt::on);
        plt::plot(x, plt::transform(y, [](auto y) { return -y; }), "--xr");
        plt::plot(x, plt::transform(x, [](auto x) { return x / plt::pi - 1.; }), "-:gs");
        plt::plot({1.0, 0.7, 0.4, 0.0, -0.4, -0.7, -1}, "k");
        // plt::show();
        plt::save("lineplot_sample.png");
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}