/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Pierre de Buyl (KU Leuven)
                        Rochus Schmid (RUB)
                        Note: this is a rip off of Pierre de Buyl's h5md dump
                              to write hdf5 based pdlp files. Thanks to Pierre for the clear code!
------------------------------------------------------------------------- */

/* This is an experiment .. first we get rid of everything and only write positions in the default interval
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include "hdf5.h"
#include "dump_pdlp.h"
#include "domain.h"
#include "atom.h"
#include "update.h"
#include "group.h"
#include "output.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "version.h"

using namespace LAMMPS_NS;

#define MYMIN(a,b) ((a) < (b) ? (a) : (b))
#define MYMAX(a,b) ((a) > (b) ? (a) : (b))

/** Scan common options for the dump elements
 */
static int element_args(int narg, char **arg, int *every)
{
  int iarg=0;
  while (iarg<narg) {
    if (strcmp(arg[iarg], "every")==0) {
      if (narg<2) return -1;
      *every = atoi(arg[iarg+1]);
      iarg+=2;
    } else {
      break;
    }
  }
  return iarg;
}

/* ---------------------------------------------------------------------- */

DumpPDLP::DumpPDLP(LAMMPS *lmp, int narg, char **arg) : Dump(lmp, narg, arg)
{
  if (narg<6) error->all(FLERR,"Illegal dump pdlp command");
  if (binary || compressed || multifile || multiproc)
    error->all(FLERR,"Invalid dump pdlp filename");

  if (domain->triclinic!=0)
    error->all(FLERR,"Invalid domain for dump pdlp. Only orthorombic domains supported.");

  size_one = 6;
  sort_flag = 1;
  sortcol = 0;
  format_default = NULL;
  flush_flag = 0;
  unwrap_flag = 0;
  stage_name=NULL;

  every_dump = force->inumeric(FLERR,arg[3]);
  
  every_xyz = -1;
  every_image = -1;
  every_vel = -1;
  every_forces = -1;
  every_charges = -1;
  every_cell = -1;

  int iarg=5;
  int n_parsed, default_every;
  size_one=0;
  if (every_dump==0) default_every=0; else default_every=1;

  while (iarg<narg) {
    if (strcmp(arg[iarg], "xyz")==0) {
      every_xyz=default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_xyz);
      if (n_parsed<0) error->all(FLERR, "Illegal dump pdlp command");
      iarg += n_parsed;
      size_one+=domain->dimension;
    } else if (strcmp(arg[iarg], "stage")==0) {
      if (iarg+1>=narg) {
        error->all(FLERR, "Invalid number of arguments in dump pdlp");
      }
      if (stage_name==NULL) {
        stage_name = new char[strlen(arg[iarg])+1];
        strcpy(stage_name, arg[iarg+1]);
      } else {
        error->all(FLERR, "Illegal dump pdlp command: stage name argument repeated");
      }
      iarg+=2;
    } else if (strcmp(arg[iarg], "xyz_img")==0) {
      if (every_xyz<0) error->all(FLERR, "Illegal dump pdlp command");
      iarg+=1;
      size_one+=domain->dimension;
      every_image = every_xyz;
    } else if (strcmp(arg[iarg], "vel")==0) {
      every_vel = default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_vel);
      if (n_parsed<0) error->all(FLERR, "Illegal dump h5md command");
      iarg += n_parsed;
      size_one+=domain->dimension;
    } else if (strcmp(arg[iarg], "forces")==0) {
      every_forces = default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_forces);
      if (n_parsed<0) error->all(FLERR, "Illegal dump h5md command");
      iarg += n_parsed;
      size_one+=domain->dimension;
    } else if (strcmp(arg[iarg], "charges")==0) {
      if (!atom->q_flag)
        error->all(FLERR, "Requesting non-allocated quantity q in dump_pdlp");
      every_charges = default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_charges);
      if (n_parsed<0) error->all(FLERR, "Illegal dump pdlp command");
      iarg += n_parsed;
      size_one+=1;
    } else {
      error->all(FLERR, "Invalid argument to dump h5md");
    }
  printf("DUMP PDLP .. all arguments parsed\n");
  printf("every_xyz %d\n", every_xyz);
  printf("every_vel %d\n", every_vel);
  printf("every_forces %d\n", every_forces);
  printf("every_charges %d\n", every_charges);
  printf("every_cell %d\n", every_cell);
  }

  // allocate global array for atom coords

  bigint n = group->count(igroup);
  natoms = static_cast<int> (n);

  if (every_xyz>=0)
    memory->create(dump_xyz,domain->dimension*natoms,"dump:xyz");
  if (every_image>=0)
    memory->create(dump_img,domain->dimension*natoms,"dump:xyz_img");
  if (every_vel>=0)
    memory->create(dump_vel,domain->dimension*natoms,"dump:vel");
  if (every_forces>=0)
    memory->create(dump_forces,domain->dimension*natoms,"dump:forces");
  if (every_charges>=0)
    memory->create(dump_charges,natoms,"dump:charges");

  // RS here the file is opened .. we need to see if we can just pass the hid_t of the hdf5 file and access it
  openfile();
  ntotal = 0;
}

/* ---------------------------------------------------------------------- */

DumpPDLP::~DumpPDLP()
{
  //  needs fixing!! RS
  if (every_xyz>=0) {
    memory->destroy(dump_xyz);
    if (me==0) H5Dclose(xyz_dset);    
  }
  if (every_image>=0) {
    memory->destroy(dump_img);
    if (me==0) H5Dclose(img_dset);    
  }
  if (every_vel>=0) {
    memory->destroy(dump_vel);
    if (me==0) H5Dclose(vel_dset);    
  }
  if (every_forces>=0) {
    memory->destroy(dump_forces);
    if (me==0) H5Dclose(forces_dset);    
  }
  if (every_charges>=0) {
    memory->destroy(dump_charges);
    if (me==0) H5Dclose(charges_dset);    
  }
  if (me==0){
    H5Gclose(traj_group);
    H5Gclose(stage_group);
    H5Fclose(pdlpfile);
  }
}

/* ---------------------------------------------------------------------- */

void DumpPDLP::init_style()
{
  if (sort_flag == 0 || sortcol != 0)
    error->all(FLERR,"Dump pdlp requires sorting by atom ID");
}

/* ---------------------------------------------------------------------- */

void DumpPDLP::openfile()
{
  int dims[2];
 
  if (me == 0) {
    // me == 0 _> do only on master node
    
    pdlpfile = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    stage_group = H5Gopen(pdlpfile, stage_name, H5P_DEFAULT);
    traj_group  = H5Gopen(stage_group, "traj", H5P_DEFAULT);
    restart_group = H5Gopen(stage_group, "restart", H5P_DEFAULT);
    printf("pdlp file opened   %d %d %d %d\n", pdlpfile, stage_group, traj_group, restart_group);

    if (every_xyz>0) {
      xyz_dset    = H5Dopen(traj_group, "xyz", H5P_DEFAULT);
      printf("pdlp xyz dset opened\n");
    }
    if (every_image>0) {
      img_dset    = H5Dopen(traj_group, "imgidx", H5P_DEFAULT);
      printf("pdlp img dset opened\n");
    }
    if (every_vel>0) {
      vel_dset    = H5Dopen(traj_group, "vel", H5P_DEFAULT);
      printf("pdlp vel dset opened\n");
    }
    if (every_forces>0) {
      forces_dset = H5Dopen(traj_group, "forces", H5P_DEFAULT);
      printf("pdlp forces dset opened\n");
    }
    if (every_charges>0) {
      charges_dset = H5Dopen(traj_group, "charges", H5P_DEFAULT);
      printf("pdlp charges dset opened\n");
    }
    if (every_restart>0) {
      rest_xyz_dset = H5Dopen(restart_group, "xyz", H5P_DEFAULT);
      rest_vel_dset = H5Dopen(restart_group, "vel", H5P_DEFAULT);
      rest_cell_dset = H5Dopen(restart_group, "cell", H5P_DEFAULT);
      printf("pdlp restart dsets opened\n");
    }

    dims[0] = natoms;
    dims[1] = domain->dimension;
  }
}

/* ---------------------------------------------------------------------- */

void DumpPDLP::write_header(bigint nbig)
{
  return;
}

/* ---------------------------------------------------------------------- */

void DumpPDLP::pack(tagint *ids)
{
  int m,n;

  tagint *tag = atom->tag;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *q = atom->q;

  imageint *image = atom->image;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int dim=domain->dimension;

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  m = n = 0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (every_xyz>=0) {
        int ix = (image[i] & IMGMASK) - IMGMAX;
        int iy = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
        int iz = (image[i] >> IMG2BITS) - IMGMAX;
        if (unwrap_flag == 1) {
          buf[m++] = (x[i][0] + ix * xprd);
          buf[m++] = (x[i][1] + iy * yprd);
          if (dim>2) buf[m++] = (x[i][2] + iz * zprd);
        } else {
          buf[m++] = x[i][0];
          buf[m++] = x[i][1];
          if (dim>2) buf[m++] = x[i][2];
        }
        if (every_image>=0) {
          buf[m++] = ix;
          buf[m++] = iy;
          if (dim>2) buf[m++] = iz;
        }
      }
      if (every_vel>=0) {
        buf[m++] = v[i][0];
        buf[m++] = v[i][1];
        if (dim>2) buf[m++] = v[i][2];
      }
      if (every_forces>=0) {
        buf[m++] = f[i][0];
        buf[m++] = f[i][1];
        if (dim>2) buf[m++] = f[i][2];
      }
      ids[n++] = tag[i];
    }
}

/* ---------------------------------------------------------------------- */

void DumpPDLP::write_data(int n, double *mybuf)
{
  // copy buf atom coords into global array

  int m = 0;
  int dim = domain->dimension;
  int k = dim*ntotal;
  int k_img = dim*ntotal;
  int k_vel = dim*ntotal;
  int k_frc = dim*ntotal;
  int k_chg = ntotal;

  for (int i = 0; i < n; i++) {
    if (every_xyz>=0) {
      for (int j=0; j<dim; j++) {
        dump_xyz[k++] = mybuf[m++];
      }
      if (every_image>=0)
        for (int j=0; j<dim; j++) {
          dump_img[k_img++] = mybuf[m++];
        }
    }
    if (every_vel>=0)
      for (int j=0; j<dim; j++) {
        dump_vel[k_vel++] = mybuf[m++];
      }
    if (every_forces>=0)
      for (int j=0; j<dim; j++) {
        dump_forces[k_frc++] = mybuf[m++];
      }
    if (every_charges>=0)
      dump_charges[k_chg++] = mybuf[m++];
    ntotal++;
  }

  // if last chunk of atoms in this snapshot, write global arrays to file

  if (ntotal == natoms) {
    if (every_xyz>0) {
      write_frame();
      ntotal = 0;
    } 
    /*else {
      write_fixed_frame();
    }
    */
  }
}

/* ---------------------------------------------------------------------- */

int DumpPDLP::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"unwrap") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal dump_modify command");
    if (strcmp(arg[1],"yes") == 0) unwrap_flag = 1;
    else if (strcmp(arg[1],"no") == 0) unwrap_flag = 0;
    else error->all(FLERR,"Illegal dump_modify command");
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

void DumpPDLP::write_frame()
{
  int local_step;
  double local_time;
  double cell[9];
  int i;
  int statcode;

  local_step = update->ntimestep;
  local_time = local_step * update->dt;
  for (i=0; i<9; i++) cell[i] = 0.0;
  cell[0] = boxxhi - boxxlo;
  cell[4] = boxyhi - boxylo;
  cell[8] = boxzhi - boxzlo;
  
  if (every_xyz>0) {
    if (local_step % (every_xyz*every_dump) == 0) {
      statcode = append_data(xyz_dset, 3, dump_xyz);
    }
  }
  if (every_cell>0 && local_step % (every_cell*every_dump) == 0) {
    statcode = append_data(cell_dset, 3, dump_cell);
  }  
  if (every_vel>0 && local_step % (every_vel*every_dump) == 0) {
    statcode = append_data(vel_dset, 3, dump_vel);
  }
  if (every_forces>0 && local_step % (every_forces*every_dump) == 0) {
    statcode = append_data(forces_dset, 3, dump_forces);
  }
  if (every_charges>0 && local_step % (every_charges*every_dump) == 0) {
    statcode = append_data(charges_dset, 2, dump_charges);
  }
}

int DumpPDLP::append_data(hid_t dset, int rank, double *dump)
{
  herr_t  status;
  hsize_t dims[rank], start[rank], count[rank];
  hid_t   fspace, mspace;
  int i;
  
  fspace = H5Dget_space(dset);
  // get current dims
  H5Sget_simple_extent_dims(fspace, dims, NULL);
  // increment by one frame
  dims[0] += 1;
  status = H5Dset_extent(dset, dims);
  H5Sclose(fspace);
  if (status<0){
    printf("Extending pdlp dataset went wrong! status is %d\n", status);
    return -1;
  }
  // Now get fspace again
  fspace = H5Dget_space(xyz_dset);
  // create start and offset
  start[0] = dims[0]-1;
  count[0] = 1;
  for (i=1; i<rank; i++) {
    start[i] = 0;
    count[i] = dims[i];
  }
  // select part of file to be writen
  status = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
  if (status<0){
    printf("Selecting hyperslab went wrong! status is %d\n", status);
    H5Sclose(fspace);
    return -2;
  }
  // generate a mspace for the data in memory
  mspace = H5Screate_simple(rank-1, dims+1, NULL);
  // write the data
  status = H5Dwrite(dset, H5T_IEEE_F64LE, mspace, fspace, H5P_DEFAULT, dump);
  // close selections
  H5Sclose(fspace);
  H5Sclose(mspace);
  if (status<0){
    printf("Writing data went wrong! status is %d\n", status);
    return -3;
  }      
return 0;
}
