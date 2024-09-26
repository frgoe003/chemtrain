#ifndef main_h
#define main_h

#include <vector>
#include <memory>
#include <string>

namespace jcn {

    struct ConnectorConfig{
        std::string mlir_module;


        std::string neighbor_list_type;

        std::string backend;
        int device;
    };

    class Connector {
    public:
        Connector(ConnectorConfig config);
        ~Connector();

        double compute_force(int inum, int gnum, double **x, double** f, int *type, int *ilist,
            int *numneigh, int **firstneigh);

    private:
        class Impl; // Forward declaration of the implementation class
        std::unique_ptr<Impl> impl_; // Use unique_ptr to manage the implementation
    };
}

#endif
