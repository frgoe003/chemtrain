//
// Created by Paul Fuchs on 30.09.24.
//

#include <chrono>

#include "connector/domain.h"
#include "connector/pjrt.h"
#include "connector/utils.h"

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

            xla::Shape count_shape = xla::ShapeUtil::MakeShape(
               xla::S32, absl::Span<const int64_t>{1,});

            position_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(position_shape));
            species_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(species_shape));
            locals_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(count_shape));
            ghosts_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(count_shape));
        }

        return AtomShapes{max_atoms, reallocate};

	}

    std::vector<xla::PjRtBuffer*> AtomBuilder::build_domain(xla::PjRtClient* client, int device_id, int inum, int gnum, double **x, int *type) {

        // If number of atoms in domain (including ghost) exceeds the allocated
        // buffers

        Logger logger = Logger::getlogger();

        auto start = std::chrono::high_resolution_clock::now();

        if (!position_literal || (inum + gnum) > max_atoms) {
            throw std::runtime_error("Domain not initialized or too many atoms");
        }

        if (logger.log(LogLevel::DEBUG)) {
            logger.log(LogLevel::DEBUG, "Positions: " + position_literal->ToString());
            logger.log(LogLevel::DEBUG, "Species: " + species_literal->ToString());
        }

        float *position_data = position_literal->data<float>().data();
        int *species_data = species_literal->data<int>().data();

        // Collect data for all local atoms and ghost atoms
        for (int i = 0; i < inum + gnum; i++) {
            std::transform(x[i], x[i] + 3, position_data + i * 3, [](double t) { return static_cast<float>(t); });
        }
        std::fill(position_data + (inum + gnum) * 3, position_data + max_atoms * 3, 0.0f);

        // Adjust species values
        std::transform(type, type + inum + gnum, species_data, [](int t) { return t - 1; });
        std::fill(species_data + inum + gnum, species_data + max_atoms, 0);

        // Provide information about number of ghost and number of local atoms
        locals_literal->data<int>().data()[0] = inum;
        ghosts_literal->data<int>().data()[0] = gnum;


        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        logger.log(LogLevel::DEBUG, "Time taken for atom literal creation: " + std::to_string(duration.count()) + " seconds");

        // Create the buffers
     	// TODO: Maybe explicit deallocation is required

        buffers.clear();
        buffers.push_back(create_buffer(client, device_id, position_literal.get()));
        buffers.push_back(create_buffer(client, device_id, species_literal.get()));
        buffers.push_back(create_buffer(client, device_id, locals_literal.get()));
        buffers.push_back(create_buffer(client, device_id, ghosts_literal.get()));

        std::vector<xla::PjRtBuffer*> buffer_ptrs;
        for (int i = 0; i < buffers.size(); i++) {
            buffer_ptrs.push_back(buffers[i].get());
        }

        return buffer_ptrs;

    }

    double AtomBuilder::evaluate_domain(bool success, int inum, double **f, std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>& results) {

        double potential;

        Logger logger = Logger::getlogger();

        if (success) {
            auto start = std::chrono::high_resolution_clock::now();

            absl::StatusOr<std::shared_ptr<xla::Literal>> force_literal = results[0][0]->ToLiteralSync();
            absl::StatusOr<std::shared_ptr<xla::Literal>> energy_literal = results[0][1]->ToLiteralSync();

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

            if (logger.log(LogLevel::DEBUG)) {
                logger.log(LogLevel::DEBUG, "Force literal: " + force_literal.value()->ToString());
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            logger.log(LogLevel::DEBUG, "Time taken for force backtransfer: " + std::to_string(duration.count()) + " seconds");

            potential = static_cast<double>(potential_data[0]);

            // Remove the buffers after computation
            buffers.clear();

        }

        return potential;

    }

} // namespace jcn
