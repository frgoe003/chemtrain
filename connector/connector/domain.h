//
// Created by Paul Fuchs on 30.09.24.
//

#include "xla/literal_util.h"

#ifndef DOMAIN_H
#define DOMAIN_H

namespace jcn {

    struct Atoms {
        // This information is required to refine the shapes of the energy
        // model
        int n_atoms;

        // The atom constructor keeps track wheter a recompilation of the
        // mlir module is necessary
        bool reallocate;

        // These are pointers to the actual data
        xla::Literal* positions;
        xla::Literal* species;
        xla::Literal* ghost_mask;

    };

    class AtomBuilder {
    public:
        AtomBuilder(float atom_multiplier) : max_atoms(0), atom_multiplier(atom_multiplier) {};
        ~AtomBuilder() = default;

        // Padds the atom data to reduce number of recompilations
        Atoms build_domain(int inum, int gnum, double **x, int *type);

        // Writes back the force to the original array and returns the potential
        double evaluate_domain(int inum, double **f, std::shared_ptr<xla::Literal> forces, std::shared_ptr<xla::Literal> potential);

    private:
        int max_atoms;

        std::unique_ptr<xla::Literal> position_literal;
        std::unique_ptr<xla::Literal> species_literal;
        std::unique_ptr<xla::Literal> ghosts_literal;

        float atom_multiplier;

    };

} // namespace jcn

#endif //DOMAIN_H
