/* ----------------------------------------------------------------------------
    chemtrain-deploy - LAMMPS plugin
    Copyright (C) 2025  Multiscale Modeling of Fluid Materials, TU Munich

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See the LICENSE file in the directory of this file.
---------------------------------------------------------------------------- */
#include "pair_chemtrain_deploy.h"

#include "libconnector.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdlib.h>

#if defined(__linux__)
#include <dlfcn.h>
#endif

using namespace LAMMPS_NS;

namespace {

class ProfileRange {
 public:
  explicit ProfileRange(const char *name) {
#if defined(__linux__)
    // Profiling is opt-in: resolving NVTX on every communication callback is
    // useful under Nsight but is not part of the production data path.
    static const bool enabled = std::getenv("JCN_COMM_PROFILE") != nullptr;
    if (!enabled) return;
    using Push = int (*)(const char *);
    static auto push = reinterpret_cast<Push>(
        dlsym(RTLD_DEFAULT, "nvtxRangePushA"));
    if (push != nullptr) {
      push(name);
      active_ = true;
    }
#else
    (void) name;
#endif
  }

  ~ProfileRange() {
#if defined(__linux__)
    if (!active_) return;
    using Pop = int (*)();
    static auto pop = reinterpret_cast<Pop>(dlsym(RTLD_DEFAULT,
                                                  "nvtxRangePop"));
    if (pop != nullptr) pop();
#endif
  }

 private:
  bool active_ = false;
};

bool communication_debug_enabled() {
  static const bool enabled = std::getenv("JCN_COMM_DEBUG") != nullptr;
  return enabled;
}

double debug_first_value(void *data,
                         jcn::CommunicationScalarType type,
                         std::int64_t rows,
                         std::int64_t cols) {
  if (data == nullptr || rows <= 0 || cols <= 0) return 0.0;

  if (type == jcn::CommunicationScalarType::F32) {
    return static_cast<double>(static_cast<float *>(data)[0]);
  }

  return static_cast<double *>(data)[0];
}

int checked_communication_count(int n, std::int64_t cols) {
  if (n < 0 || cols <= 0 ||
      static_cast<std::int64_t>(n) >
          std::numeric_limits<int>::max() / cols) {
    throw std::runtime_error(
        "LAMMPS communication buffer count exceeds the integer ABI limit");
  }
  return static_cast<int>(static_cast<std::int64_t>(n) * cols);
}

}  // namespace

/* ---------------------------------------------------------------------- */

ChemtrainDeploy::ChemtrainDeploy(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 0;
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
}

/* ---------------------------------------------------------------------- */

ChemtrainDeploy::~ChemtrainDeploy()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(xold);
  }
}

bool ChemtrainDeploy::check_distance() {
  double **x = atom->x;
  int nlocal = atom->nlocal;

  double deltasq = 2.0 * 2.0; // Hard coded skin distance

  int flag = 0;
  double local_max_displacement_sq = 0.0;
  for (int i = 0; i < nlocal; i++) {
    double delx = x[i][0] - xold[i][0];
    double dely = x[i][1] - xold[i][1];
    double delz = x[i][2] - xold[i][2];
    double rsq = delx * delx + dely * dely + delz * delz;
    local_max_displacement_sq = std::max(local_max_displacement_sq, rsq);
    if (rsq > deltasq) {
      flag = 1;
    }
  }

  int flagall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_MAX, world);
  MPI_Allreduce(&local_max_displacement_sq, &max_displacement_sq, 1,
                MPI_DOUBLE, MPI_MAX, world);

  bool update_list = (flagall > 0);

  if (update_list) {
    for (int i = 0; i < atom->nlocal; i++) {
      std::memcpy(xold[i], atom->x[i], 3 * sizeof(double));
    }
  }

  return update_list;
}

/* ---------------------------------------------------------------------- */

void ChemtrainDeploy::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  auto start = std::chrono::high_resolution_clock::now();

  // Check if neighborlist was updated just in this timestep resulting in newly communicated atoms.
  bool moved_beyond_skin = check_distance();
  bool lammps_neighbor_rebuilt = (neighbor->ago == 0);
  bool update_list = moved_beyond_skin || lammps_neighbor_rebuilt;

  // Number of sender atoms can change depending on the ghost setting of the neighbor list.
  int inum = list->inum + list->gnum;

  int retry_flag = 0;
  jcn::Results results;

  try {
    results = connector->compute_force(
      atom->nlocal, atom->nghost, atom->x, atom->f, atom->type,
      inum, list->ilist, list->numneigh, list->firstneigh, update_list, false
    );
  } catch (const jcn::RecompilationRequired& e) {
    retry_flag = 1;
  }

  int retry_flag_all;
  MPI_Allreduce(&retry_flag, &retry_flag_all, 1, MPI_INT, MPI_MAX, world);

  // If one device must recompile, give all other devices the possibility to recompile too.
  if (retry_flag_all > 0) {
    results = connector->compute_force(
      atom->nlocal, atom->nghost, atom->x, atom->f, atom->type,
      inum, list->ilist, list->numneigh, list->firstneigh, update_list, true
    );
  }

  // Scale the forces.
  if (scale != 1.0) {
    double **f = atom->f;
    for (int i = 0; i < inum; i++) {
      f[i][0] *= scale;
      f[i][1] *= scale;
      f[i][2] *= scale;
    }
  }

  // Pass the evaluated potential energy to LAMMPS.
  if (eflag) {
    eng_vdwl = scale * results.potential;
  }

  flops += static_cast<double>(results.stats.flops);
  recompilations += static_cast<int>(results.stats.recompiled);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  (void)duration;

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void ChemtrainDeploy::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(xold, atom->nmax, 3, "pair:xold");

  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) setflag[i][j] = 0;
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void ChemtrainDeploy::settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR, "Illegal jax_connect command");

  jcn::ConnectorConfig config;
  communication_enabled = false;

  // Assign devices based on local rank.
  int device_id = 0;
  char* local_rank;

  if ((local_rank = getenv("SLURM_LOCALID"))) {
    device_id = std::stoi(local_rank);
    utils::logmesg(lmp, "Assign device based on SLURM_LOCALID");
  }
  if ((local_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK"))) {
    utils::logmesg(lmp, "Assign device based on OMPI_COMM_WORLD_LOCAL_RANK");
    device_id = std::stoi(local_rank);
  }
  if ((local_rank = getenv("MV2_COMM_WORLD_LOCAL_RANK"))) {
    utils::logmesg(lmp, "Assign device based on MV2_COMM_WORLD_LOCAL_RANK");
    device_id = std::stoi(local_rank);
  }
  if ((local_rank = getenv("FLUX_TASK_LOCAL_ID"))) {
    utils::logmesg(lmp, "Assign device based on FLUX_TASK_LOCAL_ID");
    device_id = std::stoi(local_rank);
  }
  if ((local_rank = getenv("PMI_LOCAL_RANK"))) {
    utils::logmesg(lmp, "Assign device based on PMI_LOCAL_RANK");
    device_id = std::stoi(local_rank);
  }

  config.backend = std::string(arg[0]);
  config.device = device_id;

  // Record which pre-exported model variant pair_coeff should load. The
  // executable is selected later, after the model file has been read.
  int option = 1;
  if (option < narg && std::string(arg[option]) != "comm") {
    config.memory_fraction = std::stof(arg[option++]);
  }
  while (option < narg) {
    if (std::string(arg[option]) != "comm" || option + 1 >= narg) {
      error->all(FLERR,
                 "Expected 'comm on' or 'comm off' in pair_style "
                 "chemtrain_deploy settings");
    }
    const std::string value = arg[option + 1];
    if (value == "on") {
      communication_enabled = true;
    } else if (value == "off") {
      communication_enabled = false;
    } else {
      error->all(FLERR,
                 "The chemtrain/deploy comm setting must be on or off");
    }
    option += 2;
  }

  try {
    connector = std::make_unique<jcn::Connector>(config);
  } catch (const std::exception& e) {
    std::string msg =
        std::string("chemtrain_deploy: failed to initialize connector: ") +
        e.what();
    error->all(FLERR, msg.c_str());
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void ChemtrainDeploy::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg < 4) error->all(FLERR, "Illegal jax_connect command");

  std::string exported_model_path = arg[2];

  const float atom_multiplier = std::stof(arg[3]);

  std::vector<float> neighbor_list_multipliers;
  for (int i = 4; i < narg; i++) {
    neighbor_list_multipliers.push_back(std::stof(arg[i]));
  }

  std::ifstream file(exported_model_path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + exported_model_path);
  }

  std::string exported_model((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());

  jcn::ModelConfig config;

  config.model = exported_model;
  config.neighbor_list_multipliers = neighbor_list_multipliers;
  config.atom_multiplier = atom_multiplier;
  config.newton = force->newton_pair;
  config.use_communication = communication_enabled;
  if (communication_enabled) {
    // The selected graph contains FFI gathers; bind them to LAMMPS's normal
    // forward/reverse pair communication for this pair instance.
    config.communication.context = this;
    config.communication.owned_rows = &ChemtrainDeploy::owned_rows_callback;
    config.communication.active_rows = &ChemtrainDeploy::active_rows_callback;
    config.communication.exchange = &ChemtrainDeploy::exchange_callback;
  }

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      setflag[i][j] = 1;
    }
  }

  model_properties = connector->load_model(config);

  // LAMMPS sizes pair communication buffers during initialization.
  // Reserve the exported maximum up front; changing these fields inside
  // the communication callback is too late and can overrun Comm buffers.
  comm_forward = model_properties.communication_buffer_width;
  comm_reverse = model_properties.communication_buffer_width;
  comm_reverse_off = model_properties.communication_buffer_width;

  std::string req_style = update->unit_style;
  std::string set_style = model_properties.unit_style;
  if (set_style != req_style) {
    error->all(
        FLERR,
        "The units of the model do not match the unit style {:s}. "
        "Please use the units from {:s} to {:s}.",
        req_style, set_style);
  }
}

/* ---------------------------------------------------------------------- */

int ChemtrainDeploy::exchange_callback(
    void *context, void *data, std::int64_t rows, std::int64_t cols,
    jcn::CommunicationScalarType type, bool reverse, const char **error_msg) {
  auto *self = static_cast<ChemtrainDeploy *>(context);
  try {
    return self->exchange(data, rows, cols, type, reverse);
  } catch (const std::exception &e) {
    self->communication_error = e.what();
    if (error_msg != nullptr) *error_msg = self->communication_error.c_str();
    return 1;
  }
}

std::int64_t ChemtrainDeploy::active_rows_callback(void *context) {
  auto *self = static_cast<ChemtrainDeploy *>(context);
  return static_cast<std::int64_t>(self->atom->nlocal) + self->atom->nghost;
}

std::int64_t ChemtrainDeploy::owned_rows_callback(void *context) {
  auto *self = static_cast<ChemtrainDeploy *>(context);
  return static_cast<std::int64_t>(self->atom->nlocal);
}

/* ---------------------------------------------------------------------- */

int ChemtrainDeploy::exchange(void *data, std::int64_t rows,
                              std::int64_t cols,
                              jcn::CommunicationScalarType type,
                              bool reverse) {
  const std::int64_t required =
      static_cast<std::int64_t>(atom->nlocal) + atom->nghost;

  if (communication_debug_enabled()) {
    std::cerr << "[COMM] " << (reverse ? "REV" : "FWD")
              << " rows=" << rows
              << " cols=" << cols
              << " required=" << required
              << " first_before=" << debug_first_value(data, type, rows, cols)
              << std::endl;
  }

  if (data == nullptr || cols <= 0 || rows < required) {
    throw std::runtime_error("Invalid packed feature buffer from PJRT");
  }

  if (cols > std::numeric_limits<int>::max()) {
    throw std::runtime_error("Packed feature width exceeds LAMMPS limits");
  }

  if (cols > model_properties.communication_buffer_width) {
    throw std::runtime_error(
        "Packed feature width exceeds exported communication buffer width");
  }

  communication_data = data;
  communication_rows = rows;
  communication_cols = cols;
  communication_type = type;

  if (reverse) {
    ProfileRange range("chemtrain_comm.lammps_reverse");
    // This is the transpose of the forward ghost overwrite: LAMMPS sends
    // ghost cotangents back to their owning ranks, where unpack_reverse adds
    // them to the local feature gradient.
    comm->reverse_comm(this);

    if (communication_debug_enabled()) {
      std::cerr << "[COMM] REV after_comm first="
                << debug_first_value(data, type, rows, cols)
                << std::endl;
    }

    // After reverse accumulation, ghost cotangents should not contribute further
    // on this rank.
    ProfileRange zero_range("chemtrain_comm.zero_ghosts");
    const std::int64_t begin = static_cast<std::int64_t>(atom->nlocal) * cols;
    const std::int64_t end =
        static_cast<std::int64_t>(atom->nlocal + atom->nghost) * cols;
    if (communication_type == jcn::CommunicationScalarType::F64) {
      std::fill(static_cast<double *>(communication_data) + begin,
                static_cast<double *>(communication_data) + end, 0.0);
    } else {
      std::fill(static_cast<float *>(communication_data) + begin,
                static_cast<float *>(communication_data) + end, 0.0f);
    }

    if (communication_debug_enabled()) {
      std::cerr << "[COMM] REV after_zero first="
                << debug_first_value(data, type, rows, cols)
                << std::endl;
    }
  } else {
    ProfileRange range("chemtrain_comm.lammps_forward");
    // LAMMPS owns the domain decomposition, so its standard pair exchange is
    // the source of truth for replacing each ghost feature with its owner's
    // current value at this message-passing boundary.
    comm->forward_comm(this);

    if (communication_debug_enabled()) {
      std::cerr << "[COMM] FWD after_comm first="
                << debug_first_value(data, type, rows, cols)
                << std::endl;
    }
  }

  communication_data = nullptr;
  communication_rows = 0;
  communication_cols = 0;

  return 0;
}

/* ---------------------------------------------------------------------- */

double ChemtrainDeploy::communication_get(std::int64_t row,
                                          std::int64_t col) const {
  const std::int64_t index = row * communication_cols + col;
  if (communication_type == jcn::CommunicationScalarType::F64) {
    return static_cast<double *>(communication_data)[index];
  }
  return static_cast<float *>(communication_data)[index];
}

void ChemtrainDeploy::communication_set(std::int64_t row, std::int64_t col,
                                        double value) {
  const std::int64_t index = row * communication_cols + col;
  if (communication_type == jcn::CommunicationScalarType::F64) {
    static_cast<double *>(communication_data)[index] = value;
  } else {
    static_cast<float *>(communication_data)[index] = static_cast<float>(value);
  }
}

void ChemtrainDeploy::communication_add(std::int64_t row, std::int64_t col,
                                        double value) {
  communication_set(row, col, communication_get(row, col) + value);
}

/* ---------------------------------------------------------------------- */

int ChemtrainDeploy::pack_forward_comm(int n, int *list, double *buf,
                                       int, int *) {
  ProfileRange range("chemtrain_comm.pack_forward");
  if (communication_debug_enabled()) {
    std::cerr << "[COMM] pack_forward n=" << n
              << " cols=" << communication_cols << std::endl;
  }

  // LAMMPS owns double-precision communication buffers. Preserve a bulk-copy
  // path for f64 features and convert f32 only at this API boundary, keeping
  // the model's packed representation unchanged everywhere else.
  const int count = checked_communication_count(n, communication_cols);
  const int width = static_cast<int>(communication_cols);
  int m = 0;
  if (communication_type == jcn::CommunicationScalarType::F64) {
    const auto *data = static_cast<const double *>(communication_data);
    for (int i = 0; i < n; ++i) {
      const double *row = data +
          static_cast<std::int64_t>(list[i]) * communication_cols;
      std::copy_n(row, width, buf + m);
      m += width;
    }
  } else {
    const auto *data = static_cast<const float *>(communication_data);
    for (int i = 0; i < n; ++i) {
      const float *row = data +
          static_cast<std::int64_t>(list[i]) * communication_cols;
      for (std::int64_t j = 0; j < communication_cols; ++j) {
        buf[m++] = static_cast<double>(row[j]);
      }
    }
  }
  if (m != count) {
    throw std::runtime_error("LAMMPS forward pack count is inconsistent");
  }
  return m;
}

void ChemtrainDeploy::unpack_forward_comm(int n, int first, double *buf) {
  ProfileRange range("chemtrain_comm.unpack_forward");
  const bool debug = communication_debug_enabled();
  if (debug) {
    std::cerr << "[COMM] unpack_forward n=" << n
              << " first=" << first
              << " cols=" << communication_cols;
  }

  if (debug && n > 0 && communication_cols > 0) {
    std::cerr << " before=" << communication_get(first, 0)
              << " incoming=" << buf[0];
  }

  const int count = checked_communication_count(n, communication_cols);
  const std::int64_t offset =
      static_cast<std::int64_t>(first) * communication_cols;
  int m = 0;
  if (communication_type == jcn::CommunicationScalarType::F64) {
    std::copy_n(buf, count,
                static_cast<double *>(communication_data) + offset);
    m = count;
  } else {
    float *destination = static_cast<float *>(communication_data) + offset;
    for (std::int64_t j = 0; j < count; ++j) {
      destination[j] = static_cast<float>(buf[m++]);
    }
  }

  if (debug && n > 0 && communication_cols > 0) {
    std::cerr << " after=" << communication_get(first, 0);
  }

  if (debug) std::cerr << std::endl;
}

static double max_abs_double_buf(const double *buf, int n) {
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    m = std::max(m, std::abs(buf[i]));
  }
  return m;
}


int ChemtrainDeploy::pack_reverse_comm(int n, int first, double *buf) {
  ProfileRange range("chemtrain_comm.pack_reverse");
  const int count = checked_communication_count(n, communication_cols);
  const std::int64_t offset =
      static_cast<std::int64_t>(first) * communication_cols;
  int m = 0;
  if (communication_type == jcn::CommunicationScalarType::F64) {
    std::copy_n(static_cast<const double *>(communication_data) + offset,
                count, buf);
    m = count;
  } else {
    const float *source = static_cast<const float *>(communication_data) + offset;
    for (std::int64_t j = 0; j < count; ++j) {
      buf[m++] = static_cast<double>(source[j]);
    }
  }

  if (communication_debug_enabled()) {
    std::cerr << "[COMM] pack_reverse n=" << n
              << " first=" << first
              << " cols=" << communication_cols
              << " first_val=" << (m > 0 ? buf[0] : 0.0)
              << " maxabs=" << max_abs_double_buf(buf, m)
              << std::endl;
  }

  return m;
}


void ChemtrainDeploy::unpack_reverse_comm(int n, int *list, double *buf) {
  ProfileRange range("chemtrain_comm.unpack_reverse");
  const int count = checked_communication_count(n, communication_cols);
  int first_atom = (n > 0 ? list[0] : -1);
  const bool debug = communication_debug_enabled();
  double before = 0.0;
  if (debug && n > 0 && communication_cols > 0) {
    before = communication_get(first_atom, 0);
  }

  int m = 0;
  if (communication_type == jcn::CommunicationScalarType::F64) {
    auto *data = static_cast<double *>(communication_data);
    for (int i = 0; i < n; ++i) {
      double *row = data +
          static_cast<std::int64_t>(list[i]) * communication_cols;
      for (std::int64_t j = 0; j < communication_cols; ++j) {
        row[j] += buf[m++];
      }
    }
  } else {
    auto *data = static_cast<float *>(communication_data);
    for (int i = 0; i < n; ++i) {
      float *row = data +
          static_cast<std::int64_t>(list[i]) * communication_cols;
      for (std::int64_t j = 0; j < communication_cols; ++j) {
        row[j] += static_cast<float>(buf[m++]);
      }
    }
  }
  if (m != count) {
    throw std::runtime_error("LAMMPS reverse unpack count is inconsistent");
  }

  double after = 0.0;
  if (debug && n > 0 && communication_cols > 0) {
    after = communication_get(first_atom, 0);
  }

  if (debug) {
    std::cerr << "[COMM] unpack_reverse n=" << n
              << " cols=" << communication_cols
              << " target=" << first_atom
              << " first_in=" << (m > 0 ? buf[0] : 0.0)
              << " maxabs_in=" << max_abs_double_buf(buf, m)
              << " before=" << before
              << " after=" << after
              << std::endl;
  }
}

/* ---------------------------------------------------------------------- */

void ChemtrainDeploy::init_style()
{
  recompilations = 0;
  flops = 0;

  // The exported model is authoritative for the halo depth. Include the
  // LAMMPS neighbor skin so users do not need a matching comm_modify command.
  comm->cutghostuser = model_properties.comm_dist + neighbor->skin;

  int request = NeighConst::REQ_DEFAULT;

  if (model_properties.neighbor_list.include_ghosts) {
    request |= NeighConst::REQ_GHOST;
  }

  if (!model_properties.neighbor_list.half_list || force->newton) {
    // It seems like setting newton to true requires a full list.
    request |= NeighConst::REQ_FULL;
  }

  neighbor->add_request(this, request);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double ChemtrainDeploy::init_one(int i, int j)
{
  if (!allocated) allocate();

  if (setflag[i][j] == 0) error->all(FLERR, "Not all pair coeffs are set");

  for (int i = 0; i < atom->nlocal; i++) {
    std::memcpy(xold[i], atom->x[i], 3 * sizeof(double));
  }

  double min_comm_dist = model_properties.comm_dist + neighbor->skin;
  if (min_comm_dist > comm->get_comm_cutoff()) {
    error->all(
      FLERR, "Communication cutoff is too small for the model. Increase "
      "the communication cutoff to at least {:.4f}.", min_comm_dist
    );
  }

  return model_properties.cutoff;
}

/* ---------------------------------------------------------------------- */

void ChemtrainDeploy::finish()
{
  int min_comp, max_comp, sum_comp, num_procs;
  double min_flops, max_flops, sum_flops;

  MPI_Allreduce(&recompilations, &min_comp, 1, MPI_INT, MPI_MIN, world);
  MPI_Allreduce(&recompilations, &max_comp, 1, MPI_INT, MPI_MAX, world);
  MPI_Allreduce(&recompilations, &sum_comp, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&flops, &min_flops, 1, MPI_DOUBLE, MPI_MIN, world);
  MPI_Allreduce(&flops, &max_flops, 1, MPI_DOUBLE, MPI_MAX, world);
  MPI_Allreduce(&flops, &sum_flops, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Comm_size(world, &num_procs);

  double avg_comp = static_cast<double>(sum_comp) / static_cast<double>(num_procs);
  double avg_flops = sum_flops / static_cast<double>(num_procs);

  utils::logmesg(
      lmp, "\n==== JaxConnect Summary =========.\n"
           "- Recompilations: {:d} min / {:.2f} avg / {:d} max. / {:d} total \n"
           "- Estimated FLOP: {:.2e} min / {:.2e} avg / {:.2e} max. / {:.2e total\n\n",
      min_comp, avg_comp, max_comp, sum_comp,
      min_flops, avg_flops, max_flops, sum_flops);
}

/* ---------------------------------------------------------------------- */

void *ChemtrainDeploy::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "scale") == 0) return (void *) &scale;
  return nullptr;
}
