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

#include "../../connector/libconnector.h"

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

  // No allocation means no destruction

  memory->destroy(setflag);
  memory->destroy(cutsq);
  memory->destroy(cut);
  //    memory->destroy(d0);
  //    memory->destroy(alpha);
  //    memory->destroy(r0);
  //    memory->destroy(morse1);
  //    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void JaxConnect::compute(int eflag, int vflag)
{

  // The interesting stuff goes on in this function. We need to transform the neighbor list
  // in the exepected formations and also the positions into the expected format.

  // std::cout << "Start computing forces" << std::endl;

  ev_init(eflag, vflag);

  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "Neighborlist creation history: " << neighbor->ago << std::endl;

  double potential = connector->compute_force(
      atom->nlocal, atom->nghost, atom->x, atom->f, atom->type, list->ilist, list->numneigh,
      list->firstneigh
  );

  // Pass the evaluated potential energy to LAMMPS
  if (eflag) {
    eng_vdwl = potential;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Computed potential " + std::to_string(potential) << " in " << duration.count() << " seconds with JAX connector." << std::endl;


//  int i, j, ii, jj, inum, jnum, itype, jtype;
//  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
//  double rsq, r, dr, dexp, factor_lj;
//  int *ilist, *jlist, *numneigh, **firstneigh;
//
//  evdwl = 0.0;
//  ev_init(eflag, vflag);
//
//  double **x = atom->x;
//  double **f = atom->f;
//  int *type = atom->type;
//  int nlocal = atom->nlocal;
//  double *special_lj = force->special_lj;
//  int newton_pair = force->newton_pair;
//
//  inum = list->inum;
//  ilist = list->ilist;
//  numneigh = list->numneigh;
//  firstneigh = list->firstneigh;
//
//  // loop over neighbors of my atoms
//
//  for (ii = 0; ii < inum; ii++) {
//    i = ilist[ii];
//    xtmp = x[i][0];
//    ytmp = x[i][1];
//    ztmp = x[i][2];
//    itype = type[i];
//    jlist = firstneigh[i];
//    jnum = numneigh[i];
//
//    for (jj = 0; jj < jnum; jj++) {
//      j = jlist[jj];
//      factor_lj = special_lj[sbmask(j)];
//      j &= NEIGHMASK;
//
//      delx = xtmp - x[j][0];
//      dely = ytmp - x[j][1];
//      delz = ztmp - x[j][2];
//      rsq = delx * delx + dely * dely + delz * delz;
//      jtype = type[j];
//
//      if (rsq < cutsq[itype][jtype]) {
//        r = sqrt(rsq);
//        dr = r - r0[itype][jtype];
//        dexp = exp(-alpha[itype][jtype] * dr);
//        fpair = factor_lj * morse1[itype][jtype] * (dexp * dexp - dexp) / r;
//
//        f[i][0] += delx * fpair;
//        f[i][1] += dely * fpair;
//        f[i][2] += delz * fpair;
//        if (newton_pair || j < nlocal) {
//          f[j][0] -= delx * fpair;
//          f[j][1] -= dely * fpair;
//          f[j][2] -= delz * fpair;
//        }
//
//        if (eflag) {
//          evdwl = d0[itype][jtype] * (dexp * dexp - 2.0 * dexp) - offset[itype][jtype];
//          evdwl *= factor_lj;
//        }
//
//        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
//      }
//    }
//  }

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

  // No need to create something for now
  //
  // memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cut, n + 1, n + 1, "pair:cut");


  for (int i = 1; i <= n; i++) {
      for (int j = 1; j <= n; j++) setflag[i][j] = 0;
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void JaxConnect::settings(int narg, char **arg)
{
//  if (narg != 1) error->all(FLERR, "Illegal pair_style command");
//
//  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
//
//  // reset cutoffs that have been explicitly set
//
//  if (allocated) {
//    int i, j;
//    for (i = 1; i <= atom->ntypes; i++)
//      for (j = i; j <= atom->ntypes; j++)
//        if (setflag[i][j]) cut[i][j] = cut_global;
//  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void JaxConnect::coeff(int narg, char **arg)
{
  	std::cout << "Start setting coefficients after checking allocation: " << allocated << std::endl;

	if (!allocated) allocate();

    std::cout << "Allocation is now set to: " << allocated << std::endl;


    std::cout << "Number of arguments: " << narg << std::endl;
    for (int i = 0; i < narg; ++i) {
        std::cout << "arg[" << i << "]: " << arg[i] << std::endl;
    }

    const int max_neighbors = std::stoi(arg[2]);
    std::string hlo_filename = arg[3];

    std::cout << "Initialize JAX connector on module " << hlo_filename << " with " << max_neighbors << " neighbors" << std::endl;

    std::ifstream file(hlo_filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + hlo_filename);
    }

    std::string mlir_module((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::cout << "Read MLIR module with " << mlir_module.size() << " bytes" << std::endl;

    jcn::ConnectorConfig config{mlir_module, "NONE", "CUDA", 0};

    connector = std::make_unique<jcn::Connector>(config);

//  if (narg < 5 || narg > 6) error->all(FLERR, "Incorrect args for pair coefficients");
//
	int ilo, ihi, jlo, jhi;
	utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
	utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

    int count = 0;
	for (int i = ilo; i <= ihi; i++) {
    	for (int j = MAX(jlo, i); j <= jhi; j++) {
      		setflag[i][j] = 1;
            cutsq[i][j] = 25.;
            cut[i][j] = 5.;
      		count++;
    	}
  	}


//
//  double d0_one = utils::numeric(FLERR, arg[2], false, lmp);
//  double alpha_one = utils::numeric(FLERR, arg[3], false, lmp);
//  double r0_one = utils::numeric(FLERR, arg[4], false, lmp);
//
//  double cut_one = cut_global;
//  if (narg == 6) cut_one = utils::numeric(FLERR, arg[5], false, lmp);
//
//  int count = 0;
//  for (int i = ilo; i <= ihi; i++) {
//    for (int j = MAX(jlo, i); j <= jhi; j++) {
//      d0[i][j] = d0_one;
//      alpha[i][j] = alpha_one;
//      r0[i][j] = r0_one;
//      cut[i][j] = cut_one;
//      setflag[i][j] = 1;
//      count++;
//    }
//  }
//

  std::cout << "Finished allocation" << std::endl;

//  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}


void JaxConnect::init_style()
{
  // need a full neighbor list
  // TODO: Request full neighbor list
  neighbor->add_request(this,NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double JaxConnect::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

//  morse1[i][j] = 2.0 * d0[i][j] * alpha[i][j];
//
//  if (offset_flag) {
//    double alpha_dr = -alpha[i][j] * (cut[i][j] - r0[i][j]);
//    offset[i][j] = d0[i][j] * (exp(2.0 * alpha_dr) - 2.0 * exp(alpha_dr));
//  } else
//    offset[i][j] = 0.0;
//
//  d0[j][i] = d0[i][j];
//  alpha[j][i] = alpha[i][j];
//  r0[j][i] = r0[i][j];
//  morse1[j][i] = morse1[i][j];
//  offset[j][i] = offset[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

//void JaxConnect::write_restart(FILE *fp)
//{
//  write_restart_settings(fp);
//
//  int i, j;
//  for (i = 1; i <= atom->ntypes; i++)
//    for (j = i; j <= atom->ntypes; j++) {
//      fwrite(&setflag[i][j], sizeof(int), 1, fp);
//      if (setflag[i][j]) {
//        fwrite(&d0[i][j], sizeof(double), 1, fp);
//        fwrite(&alpha[i][j], sizeof(double), 1, fp);
//        fwrite(&r0[i][j], sizeof(double), 1, fp);
//        fwrite(&cut[i][j], sizeof(double), 1, fp);
//      }
//    }
//}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

//void JaxConnect::read_restart(FILE *fp)
//{
//  read_restart_settings(fp);
//
//  allocate();
//
//  int i, j;
//  int me = comm->me;
//  for (i = 1; i <= atom->ntypes; i++)
//    for (j = i; j <= atom->ntypes; j++) {
//      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
//      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
//      if (setflag[i][j]) {
//        if (me == 0) {
//          utils::sfread(FLERR, &d0[i][j], sizeof(double), 1, fp, nullptr, error);
//          utils::sfread(FLERR, &alpha[i][j], sizeof(double), 1, fp, nullptr, error);
//          utils::sfread(FLERR, &r0[i][j], sizeof(double), 1, fp, nullptr, error);
//          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
//        }
//        MPI_Bcast(&d0[i][j], 1, MPI_DOUBLE, 0, world);
//        MPI_Bcast(&alpha[i][j], 1, MPI_DOUBLE, 0, world);
//        MPI_Bcast(&r0[i][j], 1, MPI_DOUBLE, 0, world);
//        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
//      }
//    }
//}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

//void JaxConnect::write_restart_settings(FILE *fp)
//{
//  fwrite(&cut_global, sizeof(double), 1, fp);
//  fwrite(&offset_flag, sizeof(int), 1, fp);
//  fwrite(&mix_flag, sizeof(int), 1, fp);
//}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

//void JaxConnect::read_restart_settings(FILE *fp)
//{
//  if (comm->me == 0) {
//    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
//    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
//    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
//  }
//  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
//  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
//  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
//}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

//void JaxConnect::write_data(FILE *fp)
//{
//  for (int i = 1; i <= atom->ntypes; i++)
//    fprintf(fp, "%d %g %g %g\n", i, d0[i][i], alpha[i][i], r0[i][i]);
//}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

//void JaxConnect::write_data_all(FILE *fp)
//{
//  for (int i = 1; i <= atom->ntypes; i++)
//    for (int j = i; j <= atom->ntypes; j++)
//      fprintf(fp, "%d %d %g %g %g %g\n", i, j, d0[i][j], alpha[i][j], r0[i][j], cut[i][j]);
//}

/* ---------------------------------------------------------------------- */

//double JaxConnect::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
//                          double /*factor_coul*/, double factor_lj, double &fforce)
//{
//  double r, dr, dexp, phi;
//
//  r = sqrt(rsq);
//  dr = r - r0[itype][jtype];
//  dexp = exp(-alpha[itype][jtype] * dr);
//  fforce = factor_lj * morse1[itype][jtype] * (dexp * dexp - dexp) / r;
//
//  phi = d0[itype][jtype] * (dexp * dexp - 2.0 * dexp) - offset[itype][jtype];
//  return factor_lj * phi;
//}

/* ---------------------------------------------------------------------- */

//void *JaxConnect::extract(const char *str, int &dim)
//{
//  dim = 2;
////  if (strcmp(str, "d0") == 0) return (void *) d0;
////  if (strcmp(str, "r0") == 0) return (void *) r0;
////  if (strcmp(str, "alpha") == 0) return (void *) alpha;
//  return nullptr;
//}
