//
// Created by Paul Fuchs on 30.09.24.
//

#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/literal_util.h"

#ifndef DOMAIN_H
#define DOMAIN_H

namespace jcn {

	/**
     * Contains all per-atom data of a domnain
     */
    struct AtomShapes {
        /**
  		 * Maximum possible number of atoms in the domain, including local, ghost,
         * and padded atoms.
         */
        int n_atoms;

        /**
  		 * Tracks change of maxmimum number of atoms in the domain. If true,
         * must re-compile the mlir module with new shapes.
         */
        bool reallocate;

    };

    /**
     * Transforms atom data from the local domain into a padded XLA-copmpatible format
     */
    class AtomBuilder {
    public:
         /**
          * @brief Constructor
          * @param atom_multiplier Fraction of extra atoms to consider when re-allocate
    	  *         the arrays
          */
        AtomBuilder(float atom_multiplier) : max_atoms(0), atom_multiplier(atom_multiplier) {};
        ~AtomBuilder() = default;

        AtomShapes get_shapes(int inum, int gnum);

        /**
         * Padds the atom data to reduce number of recompilations
         *
         * @param inum Number of local atoms
         * @param gnum Number of ghost atoms
         * @param x Atom positions
         * @param type Atom types (one-based species)
         */
        std::vector<xla::PjRtBuffer*> build_domain(xla::PjRtClient* client, int device_id, int inum, int gnum, double **x, int *type);

        /**
         * Writes back the force to the original array and returns the potential
         *
         * @param inum Number of local atoms
         * @param f Target force array
         * @param forces Forces from the XLA computation
         * @param potential Potential from the XLA computation
         */
        double evaluate_domain(bool success, int inum, double **f, std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>& results);

    private:
        int max_atoms;

        std::unique_ptr<xla::Literal> position_literal;
        std::unique_ptr<xla::Literal> species_literal;
        std::unique_ptr<xla::Literal> locals_literal;
        std::unique_ptr<xla::Literal> ghosts_literal;

        std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers;

        float atom_multiplier;

    };

} // namespace jcn

#endif //DOMAIN_H
