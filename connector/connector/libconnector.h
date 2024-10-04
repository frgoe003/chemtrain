#ifndef main_h
#define main_h

#include <vector>
#include <memory>
#include <string>

namespace jcn {

    struct ConnectorConfig{
        // Potential module exported by jax
        std::string mlir_module;

        // String indicating the type of neighbor list expected by the program
        std::string neighbor_list_type = "SimpleSparseNeighborList";
        std::vector<float> neighbor_list_multipliers = {1.5};

        // String identifying the backend used for execution, e.g., cuda.
        std::string backend;

        // Device to run on from all addressable devices
        int device;
    };

    class Connector {
    public:
        Connector(ConnectorConfig config);
        ~Connector();


        double compute_force(int inum, int gnum, double **x, double** f, int *type, int *ilist,
            int *numneigh, int **firstneigh, bool list_changed);

    private:
        class Impl; // Forward declaration of the implementation class
        std::unique_ptr<Impl> impl_; // Use unique_ptr to manage the implementation
    };
}

#endif
