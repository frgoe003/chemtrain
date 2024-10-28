#ifndef main_h
#define main_h

#include <vector>
#include <memory>
#include <string>

namespace jcn {

    /**
     * Configurations for the evaluation of the potential
     */
    struct ConnectorConfig{
        /** Protobuffer string containing the exported model */
        std::string model;

        /**
         * Vector with multipliers for the neighbor list. The required
         * multipliers and their effect depend on the type of used neighbor list.
         */
        std::vector<float> neighbor_list_multipliers = {1.5};

        /**
         * Multiplier for reserving additional capacities for ghost atoms
         * (padding).
         */
        float atom_multiplier = 1.1;

        /**
         * String identifying the backend on which to evaluate the model.
         * The backend name is inferred from the name of the PjRt plugin file.
         */
        std::string backend;

        /**
         * TODO: Check effect
         */
        int device = 0;
    };

    /**
     * Properties of the model that can be queried by the consumer of the
     * interface, e.g., LAMMPS.
     */
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

    /**
     * Pointer to implementation class.
     */
    class Connector {
    public:
        Connector(ConnectorConfig config);
        ~Connector();

        /**
        * Computes the forces for a system given the atom positions and
        * (if required) the neighbor list.
        *
        */
        double compute_force(int inum, int gnum, double **x, double** f, int *type, int *ilist,
            int *numneigh, int **firstneigh, bool list_changed);

        ModelProperties get_model_properties();

    private:
        class Impl; // Forward declaration of the implementation class
        std::unique_ptr<Impl> impl_; // Use unique_ptr to manage the implementation
    };
}

#endif
