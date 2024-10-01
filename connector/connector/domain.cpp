//
// Created by Paul Fuchs on 30.09.24.
//

#include <chrono>

#include "domain.h"
#include "pjrt.h"

#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/literal_util.h"


namespace jcn {

	AtomShapes AtomBuilder::get_shapes(int inum, int gnum) {

        bool reallocate = false;

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

            // Destroy old literals and allocate new ones with large capacity
//            if (position_literal) {
//              	position_literal->Destroy();
//              	species_literal->Destroy();
//                ghosts_literal->Destroy();
//            }

            position_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(position_shape));
            species_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(species_shape));
            ghosts_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(ghost_shape));
        }

        return AtomShapes{max_atoms, reallocate};

	}

    std::vector<xla::PjRtBuffer*> AtomBuilder::build_domain(xla::PjRtClient* client, int device_id, int inum, int gnum, double **x, int *type) {

        // If number of atoms in domain (including ghost) exceeds the allocated
        // buffers

        auto start = std::chrono::high_resolution_clock::now();

        if (!position_literal || (inum + gnum) > max_atoms) {
            throw std::runtime_error("Domain not initialized or too many atoms");
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

        // Create the buffers
     	// TODO: Maybe explicit deallocation is required

        buffers.push_back(create_buffer(client, device_id, position_literal.get()));
        buffers.push_back(create_buffer(client, device_id, species_literal.get()));
        buffers.push_back(create_buffer(client, device_id, ghosts_literal.get()));

        std::vector<xla::PjRtBuffer*> buffer_ptrs;
        for (int i = 0; i < buffers.size(); i++) {
            buffer_ptrs.push_back(buffers[i].get());
        }

        return buffer_ptrs;

    }

    double AtomBuilder::evaluate_domain(int inum, double **f, std::vector<std::unique_ptr<xla::PjRtBuffer>> results) {
        auto start = std::chrono::high_resolution_clock::now();

        absl::StatusOr<std::shared_ptr<xla::Literal>> force_literal = results[0]->ToLiteralSync();
        absl::StatusOr<std::shared_ptr<xla::Literal>> energy_literal = results[1]->ToLiteralSync();

        if (!force_literal.ok() || !energy_literal.ok()) {
            throw std::runtime_error("Failed to convert buffer to literal");
        }

        float *force_data = force_literal.value()->data<float>().data();
        float *potential_data = energy_literal.value()->data<float>().data();

        // We skip all ghost atoms and padded atoms and only write back forces
        // on the real atoms
        for (int i = 0; i < inum; i++) {
            std::transform(force_data + 3 * i, force_data + 3 * (i + 1),
                f[i], [](float t) { return static_cast<double>(t); });
        }

        // Remove the buffers after computation
        buffers.clear();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Time taken for force backtransfer: " << duration.count() << " seconds" << std::endl;

        double potential = static_cast<double>(*potential_data);

        // Destroy the result buffers
        results.clear();

        return potential;

    }

} // namespace jcn
