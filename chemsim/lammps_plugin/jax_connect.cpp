/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
#include "jax_connect.h"

#include "libconnector.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdlib.h>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

JaxConnect::JaxConnect(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 0;
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
}

/* ---------------------------------------------------------------------- */

JaxConnect::~JaxConnect()
{
  if (allocated) {

  memory->destroy(setflag);
  memory->destroy(cutsq);
  memory->destroy(cut);
  memory->destroy(xold);

  }
}

bool JaxConnect::check_distance() {
  double **x = atom->x;
  int nlocal = atom->nlocal;

  double deltasq = 2.0 * 2.0; // Hard coded skin distance

  int flag = 0;
  for (int i = 0; i < nlocal; i++) {
    double delx = x[i][0] - xold[i][0];
    double dely = x[i][1] - xold[i][1];
    double delz = x[i][2] - xold[i][2];
    double rsq = delx * delx + dely * dely + delz * delz;
    if (rsq > deltasq) {
      flag = 1;
      break;
    }
  }

  int flagall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_MAX, world);

  bool update_list = (flagall > 0);

  if (update_list) {
    for (int i = 0; i < atom->nlocal; i++) {
      std::memcpy(xold[i], atom->x[i], 3 * sizeof(double));
    }
    std::cout << "Update positions" << std::endl;
  } else {
    std::cout << "No need to update old atom positions" << std::endl;
  }

  return update_list;

}


void JaxConnect::compute(int eflag, int vflag)
{

  ev_init(eflag, vflag);

  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "Neighborlist creation history: " << neighbor->ago << std::endl;

  // Check if neighborlist was updated just in this timestep resulting in newly communicated atoms
  bool update_list = check_distance() || (neighbor->ago == 0);

  double potential = connector->compute_force(
      atom->nlocal, atom->nghost, atom->x, atom->f, atom->type, list->ilist, list->numneigh,
      list->firstneigh, update_list
  );

  // Pass the evaluated potential energy to LAMMPS
  if (eflag) {
    eng_vdwl = potential;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Computed potential " + std::to_string(potential) << " in " << duration.count() << " seconds with JAX connector." << std::endl;

  if (vflag_fdotr) virial_fdotr_compute();

}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void JaxConnect::allocate()
{
  allocated = 1;
  std::cout << "Allocated and set flag to: " << allocated << std::endl;

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

void JaxConnect::settings(int narg, char **arg)
{
  // Settings then the pair_style command is called
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void JaxConnect::coeff(int narg, char **arg)
{

	if (!allocated) allocate();

    if (narg < 5) error->all(FLERR, "Illegal jax_connect command");

    std::string exported_model_path = arg[2];
    std::string backend = arg[3];

    const float atom_multiplier = std::stof(arg[4]);

    std::vector<float> neighbor_list_multipliers;
    for (int i = 5; i < narg; i++) {
        neighbor_list_multipliers.push_back(std::stof(arg[i]));
    }

    // Load the exported model
    std::ifstream file(exported_model_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + exported_model_path);
    }

    std::string exported_model((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    jcn::ConnectorConfig config;

    const char* nl_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");

    config.model = exported_model;
    config.neighbor_list_multipliers = neighbor_list_multipliers;
    config.atom_multiplier = atom_multiplier;
    config.backend = backend;
    config.device = 0; // std::stoi(nl_rank);

    std::cout << "Running on device: " << config.device << std::endl;
    setenv("CUDA_VISIBLE_DEVICES", nl_rank, 1);

    // Set the flags to mark initialization of all pair coefficients
    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo, i); j <= jhi; j++) {
        setflag[i][j] = 1;
      }
    }

    // Initialize the model within XLA
    connector = std::make_unique<jcn::Connector>(config);

    // We parse model properties such as the cutoff distance
    model_properties = connector->get_model_properties();

}


void JaxConnect::init_style()
{

  // Full list not required as we can simply reverse all undirected edges
  int request = NeighConst::REQ_DEFAULT;

  if (model_properties.neighbor_list.include_ghosts) {
    request |= NeighConst::REQ_GHOST;
  }
  if (!model_properties.neighbor_list.half_list) {
    request |= NeighConst::REQ_FULL;
  }

  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double JaxConnect::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  // Initialize the old atom positions
  for (int i = 0; i < atom->nlocal; i++) {
      std::memcpy(xold[i], atom->x[i], 3 * sizeof(double));
  }

  return model_properties.cutoff;
}
