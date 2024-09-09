#ifndef main_h
#define main_h

#include <vector>
#include <memory> // For std::unique_ptr

namespace jcn {
    void execute();

    class Connector {
    public:
        Connector();
        ~Connector();
        // Function declaration remains the same, but without knowing about XLA types
        std::vector<std::vector<float>> force(const std::vector<std::vector<float>>& position, const std::vector<std::vector<int>>& neighbors);

    private:
        class Impl; // Forward declaration of the implementation class
        std::unique_ptr<Impl> impl_; // Use unique_ptr to manage the implementation
    };
}

#endif
