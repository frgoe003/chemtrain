//
// Created by Paul Fuchs on 30.09.24.
//

#include <chrono>

#include "domain.h"

#include "xla/literal_util.h"


namespace jcn {

    Atoms AtomBuilder::build_domain(int inum, int gnum, double **x, int *type) {

        // If number of atoms in domain (including ghost) exceeds the allocated
        // buffers
        bool reallocate = false;

        auto start = std::chrono::high_resolution_clock::now();
        if ((inum + gnum) > max_atoms) {
            max_atoms = static_cast<int>(std::ceil(atom_multiplier * (inum + gnum)));
            reallocate = true;
        }

        // Only reallocate new memory if required
        if (!position_literal || reallocate) {
            xla::Shape position_shape = xla::ShapeUtil::MakeShape(
               xla::F32, absl::Span<const int64_t>{max_atoms, 3});
            xla::Shape species_shape = xla::ShapeUtil::MakeShape(
               xla::S32, absl::Span<const int64_t>{max_atoms,});
            xla::Shape ghost_shape = xla::ShapeUtil::MakeShape(
               xla::PRED, absl::Span<const int64_t>{max_atoms,});

            position_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(position_shape));
            species_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(species_shape));
            ghosts_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(ghost_shape));
        }

        float *position_data = position_literal->data<float>().data();
        int *species_data = species_literal->data<int>().data();
        bool *ghosts_data = ghosts_literal->data<bool>().data();

        // Collect data for all local atoms and ghost atoms
        for (int i = 0; i < inum + gnum; i++) {
            std::transform(x[i], x[i] + 3, position_data + i * 3, [](double t) { return static_cast<float>(t); });
        }
        std::fill(position_data + (inum + gnum) * 3, position_data + max_atoms * 3, 0.0f);

        // Set ghost mask for local atoms
        std::fill(ghosts_data, ghosts_data + inum, true);
        std::fill(ghosts_data + inum, ghosts_data + max_atoms, false);

        // Adjust species values
        std::transform(species_data, species_data + inum + gnum, species_data, [](int t) { return t - 1; });
        std::fill(species_data + inum + gnum, species_data + max_atoms, 0);

        std::cout << "Position of atom 3: " << position_data[3] << position_data[4] << position_data[5] << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Time taken for atom array creation: " << duration.count() << " seconds" << std::endl;

        start = std::chrono::high_resolution_clock::now();

        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        std::cout << "Time taken for atom literal creation: " << duration.count() << " seconds" << std::endl;

        return Atoms{max_atoms, reallocate, position_literal.get(), species_literal.get(), ghosts_literal.get()};

    }

    double AtomBuilder::evaluate_domain(int inum, double **f, std::shared_ptr<xla::Literal> forces, std::shared_ptr<xla::Literal> potential) {
        auto start = std::chrono::high_resolution_clock::now();

        float *force_data = forces->data<float>().data();
        absl::Span<float> potential_data = potential->data<float>();

        // We skip all ghost atoms and padded atoms and only write back forces
        // on the real atoms
        for (int i = 0; i < inum; i++) {
            std::transform(force_data + 3 * i, force_data + 3 * (i + 1),
                f[i], [](float t) { return static_cast<double>(t); });
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Time taken for force backtransfer: " << duration.count() << " seconds" << std::endl;

        return (double) potential_data[0];

    }

} // namespace jcn
