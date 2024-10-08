#ifndef main_h
#define main_h

#include <vector>
#include <memory>
#include <string>

namespace jcn {

    struct ConnectorConfig{
        // Potential module exported by jax
        std::string model;

        // String indicating the type of neighbor list expected by the program
        std::vector<float> neighbor_list_multipliers = {1.5};
        float atom_multiplier = 1.1;

        // String identifying the backend used for execution, e.g., cuda.
        std::string backend;

        // Device to run on from all addressable devices
        int device = 0;
    };

    struct ModelProperties {

        /** The cutoff distance for the potential. */
        double cutoff;

        /** Minimum distance in which ghost atoms are communicated. */
        double comm_dist = 0.0;

        struct {
            bool include_ghosts = false;
            bool half_list = true;
        } neighbor_list;

    };

    class Connector {
    public:
        Connector(ConnectorConfig config);
        ~Connector();


        double compute_force(int inum, int gnum, double **x, double** f, int *type, int *ilist,
            int *numneigh, int **firstneigh, bool list_changed);

        ModelProperties get_model_properties();

    private:
        class Impl; // Forward declaration of the implementation class
        std::unique_ptr<Impl> impl_; // Use unique_ptr to manage the implementation
    };
}

#endif
