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
   Contributing authors: Steven Vandenbrande
------------------------------------------------------------------------- */

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "fix_mttknhc.h"
#include "math_extra.h"
#include "atom.h"
#include "force.h"
#include "group.h"
#include "comm.h"
#include "neighbor.h"
#include "irregular.h"
#include "modify.h"
#include "fix_deform.h"
#include "compute.h"
#include "kspace.h"
#include "update.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "random_park.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define DELTAFLIP 0.1
#define TILTMAX 1.5
// Coefficients of Taylor expansion of sinh
#define A2 1.0/6.0
#define A4 1.0/20.0
#define A6 1.0/42.0
#define A8 1.0/72.0
#define A10 1.0/110.0
// Coefficients of Taylor expansions appearing in exp(A*t)
#define B0 1.0/72.0
#define B1 1.0/1620.0
#define B2 1.0/1080.0
#define B3 1.0/6480.0
#define B4 1.0/3240.0
#define B5 1.0/2160.0

//#define DEBUG

enum{NOBIAS,BIAS};
enum{ISO,TRICLINIC};

/* ----------------------------------------------------------------------
   MTTK barostat with NHC temperature control as implemented in YAFF
 ---------------------------------------------------------------------- */

FixMTTKNHC::FixMTTKNHC(LAMMPS *lmp, int narg, char **arg) : 
  Fix(lmp, narg, arg),
  rfix(NULL), id_dilate(NULL), irregular(NULL), id_temp(NULL), id_press(NULL),
  eta(NULL), eta_dot(NULL), eta_dotdot(NULL),
  eta_mass(NULL)
{
  if (narg < 4) error->all(FLERR,"Illegal fix nvt/npt/nph command");

  from_restart = 0;
  restart_global = 1;
  dynamic_group_allow = 1;
  time_integrate = 1;
  scalar_flag = 1;
  vector_flag = 1;
  global_freq = 1;
  extscalar = 1;
  extvector = 0;

  // default values

  allremap = 1;
  id_dilate = NULL;
  mtchain = 3;
  vol_constraint = 0;
  eta_mass_flag = 1;
  omega_mass_flag = 0;
  flipflag = 1;

  tcomputeflag = 0;
  pcomputeflag = 0;
  id_temp = NULL;
  id_press = NULL;

  dimension = domain->dimension;

  tstat_flag = 0;
  double t_period = 0.0;

  double p_period[6];
  for (int i = 0; i < 6; i++) {
    p_start[i] = p_stop[i] = p_period[i] = p_target[i] = 0.0;
    p_flag[i] = 0;
  }

  // process keywords

  int iarg = 3;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"temp") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      tstat_flag = 1;
      t_start = force->numeric(FLERR,arg[iarg+1]);
      t_target = t_start;
      t_stop = force->numeric(FLERR,arg[iarg+2]);
      t_period = force->numeric(FLERR,arg[iarg+3]);
      if (t_start <= 0.0 || t_stop <= 0.0)
        error->all(FLERR,
                   "Target temperature for fix nvt/npt/nph cannot be 0.0");
      iarg += 4;
    } else if (strcmp(arg[iarg],"iso") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      p_start[0] = p_start[1] = p_start[2] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[0] = p_stop[1] = p_stop[2] = force->numeric(FLERR,arg[iarg+2]);
      p_period[0] = p_period[1] = p_period[2] =
        force->numeric(FLERR,arg[iarg+3]);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      iarg += 4;
    } else if (strcmp(arg[iarg],"tri") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      p_start[0] = p_start[1] = p_start[2] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[0] = p_stop[1] = p_stop[2] = force->numeric(FLERR,arg[iarg+2]);
      p_period[0] = p_period[1] = p_period[2] =
        force->numeric(FLERR,arg[iarg+3]);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      p_start[3] = p_start[4] = p_start[5] = 0.0;
      p_stop[3] = p_stop[4] = p_stop[5] = 0.0;
      p_period[3] = p_period[4] = p_period[5] =
        force->numeric(FLERR,arg[iarg+3]);
      p_flag[3] = p_flag[4] = p_flag[5] = 1;
      iarg += 4;
    } else if (strcmp(arg[iarg],"tchain") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      mtchain = force->inumeric(FLERR,arg[iarg+1]);
      if (mtchain < 1) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"volconstraint") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      if (strcmp(arg[iarg+1],"yes") == 0) vol_constraint = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) vol_constraint = 0;
      else error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"flip") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      if (strcmp(arg[iarg+1],"yes") == 0) flipflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) flipflag = 0;
      else error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix nvt/npt/nph command");
  }

  // error checks

  if (dimension != 3)
    error->all(FLERR,"NHCMTTK only implemented for 3d periodic simulations");

  if (!domain->triclinic && (p_flag[3] || p_flag[4] || p_flag[5]))
    error->all(FLERR,"Can not specify Pxy/Pxz/Pyz in "
               "fix nvt/npt/nph with non-triclinic box");

  if ((tstat_flag && t_period <= 0.0) ||
      (p_flag[0] && p_period[0] <= 0.0) ||
      (p_flag[1] && p_period[1] <= 0.0) ||
      (p_flag[2] && p_period[2] <= 0.0) ||
      (p_flag[3] && p_period[3] <= 0.0) ||
      (p_flag[4] && p_period[4] <= 0.0) ||
      (p_flag[5] && p_period[5] <= 0.0))
    error->all(FLERR,"Fix nvt/npt/nph damping parameters must be > 0.0");

  // set pstat_flag and box change and restart_pbc variables

  pre_exchange_flag = 0;
  pstat_flag = 0;
  pstyle = ISO;

  for (int i = 0; i < 6; i++)
    if (p_flag[i]) pstat_flag = 1;

  if (pstat_flag) {
    if (p_flag[0] || p_flag[1] || p_flag[2]) box_change_size = 1;
    if (p_flag[3] || p_flag[4] || p_flag[5]) box_change_shape = 1;
    no_change_box = 1;
    if (allremap == 0) restart_pbc = 1;

    // pstyle = TRICLINIC if any off-diagonal term is controlled -> 6 dof
    // else pstyle = ISO -> 1 dof

    if (p_flag[3] || p_flag[4] || p_flag[5]) pstyle = TRICLINIC;

    // pre_exchange only required if flips can occur due to shape changes

    if (flipflag && (p_flag[3] || p_flag[4] || p_flag[5]))
      pre_exchange_flag = 1;
    if (flipflag && (domain->yz != 0.0 || domain->xz != 0.0 ||
                     domain->xy != 0.0))
      pre_exchange_flag = 1;
  }

  if (pstyle==ISO && vol_constraint)
    error->all(FLERR,"Constrained volume with only isotropic cell fluctuations makes no sense!"); 


  ndof_baro = 0.0;
  if (pstat_flag) {
    if (pstyle==TRICLINIC) ndof_baro = 6.0;
    else ndof_baro = 1.0;
    if (vol_constraint) ndof_baro -= 1.0;
  }

  // convert input periods to frequencies

  t_freq = 0.0;
  p_freq[0] = p_freq[1] = p_freq[2] = p_freq[3] = p_freq[4] = p_freq[5] = 0.0;

  if (tstat_flag) t_freq = 1.0 / t_period;
  if (p_flag[0]) p_freq[0] = 1.0 / p_period[0];
  if (p_flag[1]) p_freq[1] = 1.0 / p_period[1];
  if (p_flag[2]) p_freq[2] = 1.0 / p_period[2];
  if (p_flag[3]) p_freq[3] = 1.0 / p_period[3];
  if (p_flag[4]) p_freq[4] = 1.0 / p_period[4];
  if (p_flag[5]) p_freq[5] = 1.0 / p_period[5];

  // Nose/Hoover temp and pressure init

  size_vector = 0;

  if (tstat_flag) {
    int ich;
    eta = new double[mtchain];

    // add one extra dummy thermostat, set to zero

    eta_dot = new double[mtchain+1];
    eta_dot[mtchain] = 0.0;
    eta_dotdot = new double[mtchain];
    for (ich = 0; ich < mtchain; ich++) {
      eta[ich] = eta_dot[ich] = eta_dotdot[ich] = 0.0;
    }
    eta_mass = new double[mtchain];
    size_vector += 2*2*mtchain;
  }

  if (pstat_flag) {
    omega[0] = omega[1] = omega[2] = 0.0;
    omega_dot[0] = omega_dot[1] = omega_dot[2] = 0.0;
    omega_mass[0] = omega_mass[1] = omega_mass[2] = 0.0;
    omega[3] = omega[4] = omega[5] = 0.0;
    omega_dot[3] = omega_dot[4] = omega_dot[5] = 0.0;
    omega_mass[3] = omega_mass[4] = omega_mass[5] = 0.0;
    if (pstyle == ISO) size_vector += 2*2*1;
    else if (pstyle == TRICLINIC) size_vector += 2*2*6;

  }

  nrigid = 0;
  rfix = NULL;

  if (pre_exchange_flag) irregular = new Irregular(lmp);
  else irregular = NULL;

  if (!tstat_flag)
    error->all(FLERR,"Temperature control must be used with fix nhcmttk");
  if (!pstat_flag)
    error->all(FLERR,"Pressure control must be used with fix nhcmttk");

  // create a new compute temp style
  // id = fix-ID + temp
  // compute group = all since pressure is always global (group all)
  // and thus its KE/temperature contribution should use group all

  int n = strlen(id) + 6;
  id_temp = new char[n];
  strcpy(id_temp,id);
  strcat(id_temp,"_temp");

  char **newarg = new char*[3];
  newarg[0] = id_temp;
  newarg[1] = (char *) "all";
  newarg[2] = (char *) "temp";

  modify->add_compute(3,newarg);
  delete [] newarg;
  tcomputeflag = 1;

  // create a new compute pressure style
  // id = fix-ID + press, compute group = all
  // pass id_temp as 4th arg to pressure constructor

  n = strlen(id) + 7;
  id_press = new char[n];
  strcpy(id_press,id);
  strcat(id_press,"_press");

  newarg = new char*[4];
  newarg[0] = id_press;
  newarg[1] = (char *) "all";
  newarg[2] = (char *) "pressure";
  newarg[3] = id_temp;
  modify->add_compute(4,newarg);
  delete [] newarg;
  pcomputeflag = 1;

}

/* ---------------------------------------------------------------------- */

int FixMTTKNHC::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= THERMO_ENERGY;
  mask |= SECOND_INTEGRATE;
  mask |= THIRD_INTEGRATE;
  if (pre_exchange_flag) mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

FixMTTKNHC::~FixMTTKNHC()
{
  if (copymode) return;

  delete [] id_dilate;
  delete [] rfix;

  delete irregular;

  // delete temperature and pressure if fix created them

  if (tcomputeflag) modify->delete_compute(id_temp);
  delete [] id_temp;

  if (tstat_flag) {
    delete [] eta;
    delete [] eta_dot;
    delete [] eta_dotdot;
    delete [] eta_mass;
  }

  if (pstat_flag) {
    if (pcomputeflag) modify->delete_compute(id_press);
    delete [] id_press;
  }
}

/* ---------------------------------------------------------------------- */

void FixMTTKNHC::init()
{
  // recheck that dilate group has not been deleted

  if (allremap == 0) {
    int idilate = group->find(id_dilate);
    if (idilate == -1)
      error->all(FLERR,"Fix nvt/npt/nph dilate group ID does not exist");
    dilate_group_bit = group->bitmask[idilate];
  }

  // ensure no conflict with fix deform

  if (pstat_flag)
    for (int i = 0; i < modify->nfix; i++)
      if (strcmp(modify->fix[i]->style,"deform") == 0) {
        int *dimflag = ((FixDeform *) modify->fix[i])->dimflag;
        if ((p_flag[0] && dimflag[0]) || (p_flag[1] && dimflag[1]) ||
            (p_flag[2] && dimflag[2]) || (p_flag[3] && dimflag[3]) ||
            (p_flag[4] && dimflag[4]) || (p_flag[5] && dimflag[5]))
          error->all(FLERR,"Cannot use fix npt and fix deform on "
                     "same component of stress tensor");
      }

  // set temperature and pressure ptrs

  int icompute = modify->find_compute(id_temp);
  if (icompute < 0)
    error->all(FLERR,"Temperature ID for fix nvt/npt does not exist");
  temperature = modify->compute[icompute];

  if (temperature->tempbias) which = BIAS;
  else which = NOBIAS;

  if (pstat_flag) {
    icompute = modify->find_compute(id_press);
    if (icompute < 0)
      error->all(FLERR,"Pressure ID for fix npt/nph does not exist");
    pressure = modify->compute[icompute];
  }

  // set timesteps and frequencies

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dthalf = 0.5 * update->dt;
  dt4 = 0.25 * update->dt;
  dt8 = 0.125 * update->dt;
  dto = dthalf;

  p_freq_max = 0.0;
  if (pstat_flag) {
    p_freq_max = MAX(p_freq[0],p_freq[1]);
    p_freq_max = MAX(p_freq_max,p_freq[2]);
    if (pstyle == TRICLINIC) {
      p_freq_max = MAX(p_freq_max,p_freq[3]);
      p_freq_max = MAX(p_freq_max,p_freq[4]);
      p_freq_max = MAX(p_freq_max,p_freq[5]);
    }
  }

  // tally the number of dimensions that are barostatted
  // set initial volume and reference cell, if not already done

  if (pstat_flag) {
    pdim = p_flag[0] + p_flag[1] + p_flag[2];
  }

  boltz = force->boltz;
  nktv2p = force->nktv2p;

  if (force->kspace) kspace_flag = 1;
  else kspace_flag = 0;

  // detect if any rigid fixes exist so rigid bodies move when box is remapped
  // rfix[] = indices to each fix rigid

  delete [] rfix;
  nrigid = 0;
  rfix = NULL;

  for (int i = 0; i < modify->nfix; i++)
    if (modify->fix[i]->rigid_flag) nrigid++;
  if (nrigid) {
    rfix = new int[nrigid];
    nrigid = 0;
    for (int i = 0; i < modify->nfix; i++)
      if (modify->fix[i]->rigid_flag) rfix[nrigid++] = i;
  }
}

/* ----------------------------------------------------------------------
   compute T,P before integrator starts
------------------------------------------------------------------------- */

void FixMTTKNHC::setup(int vflag)
{
  RanPark *random = NULL;
  // tdof needed by compute_temp_target()
  t_current = temperature->compute_scalar();
  tdof = temperature->dof;

  // t_target is needed by NVT and NPT in compute_scalar()
  // If no thermostat or using fix nphug,
  // t_target must be defined by other means.

  if (tstat_flag) compute_temp_target();
  if (pstat_flag) compute_press_target();

  if (pstat_flag) {
    if (pstyle == ISO) pressure->compute_scalar();
    else pressure->compute_vector();
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  // masses and initial forces on thermostat variables

  if (tstat_flag) {
    eta_mass[0] = tdof * boltz * t_target / (t_freq*t_freq*4.0*M_PI*M_PI);
    for (int ich = 1; ich < mtchain; ich++) {
      eta_mass[ich] = boltz * t_target / (t_freq*t_freq*4.0*M_PI*M_PI);
    }

    if (from_restart==0){
        random = new RanPark(lmp,1);
        for (int ich = 0; ich < mtchain; ich++) {
            eta_dot[ich] = random->gaussian()*sqrt(boltz*t_target/eta_mass[ich]);
        }
        /*eta_dot[0] = -0.001205475164679217000424;
        eta_dot[1] = -0.040032364102773096425913;
        eta_dot[2] = -0.033902472744484553335287;*/
    }
    #ifdef DEBUG
    printf("%-30s","DEBUG T INIT MASS");
    for (int ich = 0; ich < mtchain; ich++) printf("%20.12f",eta_mass[ich]);
    printf("\n%-30s","DEBUG T INIT TVEL");
    for (int ich = 0; ich < mtchain; ich++) printf("%20.12f", eta_dot[ich]);
    printf("\n");
    #endif
  }

  // masses and initial forces on barostat variables

  if (pstat_flag) {
    double kt = boltz * t_target;
    double nkt = (tdof+0.5*pdim*(pdim+1))* kt;
    for (int i = 0; i < 3; i++)
      if (p_flag[i])
        {omega_mass[i] = nkt/(p_freq[i]*p_freq[i]*4.0*M_PI*M_PI);}
    if (pstyle == TRICLINIC) {
      for (int i = 3; i < 6; i++)
        if (p_flag[i]) {omega_mass[i] = nkt/(p_freq[i]*p_freq[i]*4.0*M_PI*M_PI);}
    }

    if (from_restart==0){
        random = new RanPark(lmp,1);
        if (pstyle==TRICLINIC) {
          for (int i = 0; i < 6; i++){
            omega_dot[i] = random->gaussian()*sqrt(kt/omega_mass[i]);
          }
        if (vol_constraint){
            double trace_omega_dot = (omega_dot[0] + omega_dot[1] + omega_dot[2])/3.0;
            for (int i = 0; i < 6; i++){
              omega_dot[i] -= trace_omega_dot;
            }
        }
        }
        else {
            double omega_dot_single = random->gaussian()*sqrt(kt/omega_mass[0]);
            omega_dot[0] = omega_dot_single;
            omega_dot[1] = omega_dot_single;
            omega_dot[2] = omega_dot_single;
            omega_dot[3] = 0.0;
            omega_dot[4] = 0.0;
            omega_dot[5] = 0.0;
        }
        /*omega_dot[0] = -0.000429784824523365277527;
        omega_dot[1] =  0.000002469045365480286440;
        omega_dot[2] =  0.000262877809676862785049;
        omega_dot[3] = -0.000266376040113726331431;
        omega_dot[4] = -0.000179293968948049880314;
        omega_dot[5] =  0.000445392886251490130803;*/
   }
    #ifdef DEBUG
    printf("%-30s", "DEBUG B INIT MASS");
    for (int i = 0; i < 6; i++) printf("%20.12f", omega_mass[i]);
    printf("\n%-30s", "DEBUG B INIT BVEL");
    for (int i = 0; i < 6; i++) printf("%20.12f", omega_dot[i]);
    printf("\n");
    double **v = atom->v;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    if (igroup == atom->firstgroup) nlocal = atom->nfirst;
    #endif
  }
}

/* ----------------------------------------------------------------------
   1st quarter of Verlet update
------------------------------------------------------------------------- */

void FixMTTKNHC::initial_integrate(int vflag)
{
  // need to recompute pressure to account for change in KE
  // t_current is up-to-date, but compute_temperature is not
  // compute appropriately coupled elements of mvv_current

  if (pstat_flag) {
    if (kspace_flag) force->kspace->setup();
    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
    #ifdef DEBUG
    printf("%-30s%20.12f%20.12f%20.12f%20.12f%20.12f%20.12f\n", "DEBUG F COMPUTE",p_current[0],p_current[1],p_current[2],p_current[5],p_current[4],p_current[3]);
    #endif
  }

  if (pstat_flag) {
    compute_press_target();
    nh_omega_dot();
    nh_v_press();
    if (kspace_flag) force->kspace->setup();
   }
}

/* ----------------------------------------------------------------------
   2nd quarter of Verlet update
------------------------------------------------------------------------- */

void FixMTTKNHC::second_integrate(int vflag)
{
  if (pstat_flag) {
    if (kspace_flag) force->kspace->setup();
    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_vector();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
    #ifdef DEBUG
    printf("%-30s%20.12f%20.12f%20.12f%20.12f%20.12f%20.12f\n", "DEBUG F COMPUTE",p_current[0],p_current[1],p_current[2],p_current[5],p_current[4],p_current[3]);
    #endif
    nh_omega_dot();
  }

  if (tstat_flag) {
    compute_temp_target();
    t_current = temperature->compute_scalar();
    nhc_temp_integrate();
  }
  nve_v();
  nve_x();
  if (pstat_flag) {
    if (kspace_flag) force->kspace->setup();
  }
}

/* ----------------------------------------------------------------------
   3rd quarter of Verlet update
------------------------------------------------------------------------- */

void FixMTTKNHC::third_integrate(int vflag)
{

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  nve_v();

  t_current = temperature->compute_scalar();

  if (tstat_flag) {
    if (pstat_flag) {
      if (pstyle == ISO) {
        temperature->compute_scalar();
        pressure->compute_scalar();
      } else {
        temperature->compute_vector();
        pressure->compute_vector();
      }
      couple();
    }
    compute_temp_target();
    t_current = temperature->compute_scalar();
    nhc_temp_integrate();
  }

  t_current = temperature->compute_scalar();

  if (pstat_flag) {
    if (kspace_flag) force->kspace->setup();
    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);

    if (kspace_flag) force->kspace->setup();
    compute_press_target();


    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
    #ifdef DEBUG
    printf("%-30s%20.12f%20.12f%20.12f%20.12f%20.12f%20.12f\n", "DEBUG F COMPUTE",p_current[0],p_current[1],p_current[2],p_current[5],p_current[4],p_current[3]);
    #endif

    nh_omega_dot();
    nh_v_press();
    if (kspace_flag) force->kspace->setup();

    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
    #ifdef DEBUG
    printf("%-30s%20.12f%20.12f%20.12f%20.12f%20.12f%20.12f\n", "DEBUG F COMPUTE",p_current[0],p_current[1],p_current[2],p_current[5],p_current[4],p_current[3]);
    #endif
   }
}

/* ----------------------------------------------------------------------
   4th quarter of Verlet update
------------------------------------------------------------------------- */

void FixMTTKNHC::final_integrate()
{
  if (pstat_flag) {
    if (kspace_flag) force->kspace->setup();
    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
    #ifdef DEBUG
    printf("%-30s%20.12f%20.12f%20.12f%20.12f%20.12f%20.12f\n", "DEBUG F COMPUTE",p_current[0],p_current[1],p_current[2],p_current[5],p_current[4],p_current[3]);
    #endif
    nh_omega_dot();
  }
}

/* ---------------------------------------------------------------------- */

void FixMTTKNHC::couple()
{
  double *tensor = pressure->vector;

  if (pstyle == ISO) {
    p_current[0] = p_current[1] = p_current[2] = pressure->scalar;
    if (!ISFINITE(p_current[0]) || !ISFINITE(p_current[1]) || !ISFINITE(p_current[2]))
      error->all(FLERR,"Non-numeric pressure - simulation unstable");
  }

  // switch order from xy-xz-yz to Voigt

  if (pstyle == TRICLINIC) {
    p_current[0] = tensor[0];
    p_current[1] = tensor[1];
    p_current[2] = tensor[2];
    p_current[3] = tensor[5];
    p_current[4] = tensor[4];
    p_current[5] = tensor[3];
    if (!ISFINITE(p_current[0]) || !ISFINITE(p_current[1]) || !ISFINITE(p_current[2]) || !ISFINITE(p_current[3]) || !ISFINITE(p_current[4]) || !ISFINITE(p_current[5]))
      error->all(FLERR,"Non-numeric pressure - simulation unstable");
  }
}


/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixMTTKNHC::write_restart(FILE *fp)
{
  int nsize = size_restart_global();

  double *list;
  memory->create(list,nsize,"nh:list");

  pack_restart_data(list);

  if (comm->me == 0) {
    int size = nsize * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),nsize,fp);
  }

  memory->destroy(list);
}

/* ----------------------------------------------------------------------
    calculate the number of data to be packed
------------------------------------------------------------------------- */

int FixMTTKNHC::size_restart_global()
{
  int nsize = 2;
  if (tstat_flag) nsize += 1 + 2*mtchain;
  if (pstat_flag) {
    nsize += 16;
  }

  return nsize;
}

/* ----------------------------------------------------------------------
   pack restart data
------------------------------------------------------------------------- */

int FixMTTKNHC::pack_restart_data(double *list)
{
  int n = 0;

  list[n++] = tstat_flag;
  if (tstat_flag) {
    list[n++] = mtchain;
    for (int ich = 0; ich < mtchain; ich++)
      list[n++] = eta[ich];
    for (int ich = 0; ich < mtchain; ich++)
      list[n++] = eta_dot[ich];
  }

  list[n++] = pstat_flag;
  if (pstat_flag) {
    list[n++] = omega[0];
    list[n++] = omega[1];
    list[n++] = omega[2];
    list[n++] = omega[3];
    list[n++] = omega[4];
    list[n++] = omega[5];
    list[n++] = omega_dot[0];
    list[n++] = omega_dot[1];
    list[n++] = omega_dot[2];
    list[n++] = omega_dot[3];
    list[n++] = omega_dot[4];
    list[n++] = omega_dot[5];
  }

  return n;
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixMTTKNHC::restart(char *buf)
{
  from_restart = 1;
  int n = 0;
  double *list = (double *) buf;
  int flag = static_cast<int> (list[n++]);
  if (flag) {
    int m = static_cast<int> (list[n++]);
    if (tstat_flag && m == mtchain) {
      for (int ich = 0; ich < mtchain; ich++)
        eta[ich] = list[n++];
      for (int ich = 0; ich < mtchain; ich++)
        eta_dot[ich] = list[n++];
    } else n += 2*m;
  }
  flag = static_cast<int> (list[n++]);
  if (flag) {
    omega[0] = list[n++];
    omega[1] = list[n++];
    omega[2] = list[n++];
    omega[3] = list[n++];
    omega[4] = list[n++];
    omega[5] = list[n++];
    omega_dot[0] = list[n++];
    omega_dot[1] = list[n++];
    omega_dot[2] = list[n++];
    omega_dot[3] = list[n++];
    omega_dot[4] = list[n++];
    omega_dot[5] = list[n++];
  }
}

/* ---------------------------------------------------------------------- */

int FixMTTKNHC::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    if (tcomputeflag) {
      modify->delete_compute(id_temp);
      tcomputeflag = 0;
    }
    delete [] id_temp;
    int n = strlen(arg[1]) + 1;
    id_temp = new char[n];
    strcpy(id_temp,arg[1]);

    int icompute = modify->find_compute(arg[1]);
    if (icompute < 0)
      error->all(FLERR,"Could not find fix_modify temperature ID");
    temperature = modify->compute[icompute];

    if (temperature->tempflag == 0)
      error->all(FLERR,
                 "Fix_modify temperature ID does not compute temperature");
    if (temperature->igroup != 0 && comm->me == 0)
      error->warning(FLERR,"Temperature for fix modify is not for group all");

    // reset id_temp of pressure to new temperature ID

    if (pstat_flag) {
      icompute = modify->find_compute(id_press);
      if (icompute < 0)
        error->all(FLERR,"Pressure ID for fix modify does not exist");
      modify->compute[icompute]->reset_extra_compute_fix(id_temp);
    }

    return 2;

  } else if (strcmp(arg[0],"press") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    if (!pstat_flag) error->all(FLERR,"Illegal fix_modify command");
    if (pcomputeflag) {
      modify->delete_compute(id_press);
      pcomputeflag = 0;
    }
    delete [] id_press;
    int n = strlen(arg[1]) + 1;
    id_press = new char[n];
    strcpy(id_press,arg[1]);

    int icompute = modify->find_compute(arg[1]);
    if (icompute < 0) error->all(FLERR,"Could not find fix_modify pressure ID");
    pressure = modify->compute[icompute];

    if (pressure->pressflag == 0)
      error->all(FLERR,"Fix_modify pressure ID does not compute pressure");
    return 2;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

double FixMTTKNHC::compute_scalar()
{
  int i;
  double volume;
  double energy, energy_bkin, energy_bpress;
  double kt = boltz * t_target;
  double lkt_press = kt;
  int ich;
  if (dimension == 3) volume = domain->xprd * domain->yprd * domain->zprd;
  else volume = domain->xprd * domain->yprd;

  energy = 0.0;

  if (tstat_flag) {
    energy += ke_target * eta[0] + 0.5*eta_mass[0]*eta_dot[0]*eta_dot[0];
    for (ich = 1; ich < mtchain; ich++)
      energy += kt * eta[ich] + 0.5*eta_mass[ich]*eta_dot[ich]*eta_dot[ich];
  #ifdef DEBUG
  printf("%-30s", "DEBUG T ECONS TVEL");
  for (int i=0;i<3;i++) printf("%20.12f",eta_dot[i]);
  printf("\n");
  printf("%-30s", "DEBUG T ECONS TPOS");
  for (int i=0;i<3;i++) printf("%20.12f",eta[i]);
  printf("\n");
  printf("%-30s", "DEBUG T ECONS TMAS");
  for (int i=0;i<3;i++) printf("%20.12f",eta_mass[i]);
  printf("\n");
  printf("%-30s%20.12f\n","DEBUG T ECONS",energy);
  #endif
  }

  energy_bkin = 0.0;
  energy_bpress = 0.0;
  if (pstat_flag) {
    for (i = 0; i < 3; i++)
      if (p_flag[i])
        energy_bkin += 0.5*omega_dot[i]*omega_dot[i]*omega_mass[i];
        energy_bpress += p_hydro*(volume) / (nktv2p);
    if (pstyle == TRICLINIC) {
      for (i = 3; i < 6; i++)
        if (p_flag[i])
          energy_bkin += 0.5*omega_dot[i]*omega_dot[i]*omega_mass[i];
    }
  }

  #ifdef DEBUG
  printf("%-30s%20.12f\n","DEBUG B ECONSKIN",energy_bkin);
  printf("%-30s%20.12f\n","DEBUG B ECONSPRESS",energy_bpress);
  printf("%-30s%20.12f\n","DEBUG B ECONS",energy_bkin+energy_bpress);
  #endif
  energy += energy_bkin+energy_bpress;
  energy_bkin = ndof_baro*kt*eta[0];
  energy += energy_bkin;
  #ifdef DEBUG
  printf("%-30s%20.12f\n","DEBUG TB ECONSCORR",energy_bkin);
  printf("%-30s%20.12f\n","DEBUG TB ECONS",energy);
  #endif
  return energy;
}

/* ----------------------------------------------------------------------
   return a single element of the following vectors, in this order:
      eta[tchain], eta_dot[tchain], omega[ndof], omega_dot[ndof]
      PE_eta[tchain], KE_eta_dot[tchain]
      PE_omega[ndof], KE_omega_dot[ndof]
  if no thermostat exists, related quantities are omitted from the list
  if no barostat exists, related quantities are omitted from the list
  ndof = 1,6 degrees of freedom for pstyle = ISO,TRI
------------------------------------------------------------------------- */

double FixMTTKNHC::compute_vector(int n)
{
  int ilen;

  if (tstat_flag) {
    ilen = mtchain;
    if (n < ilen) return eta[n];
    n -= ilen;
    ilen = mtchain;
    if (n < ilen) return eta_dot[n];
    n -= ilen;
  }

  if (pstat_flag) {
    if (pstyle == ISO) {
      ilen = 1;
      if (n < ilen) return omega[n];
      n -= ilen;
    } else {
      ilen = 6;
      if (n < ilen) return omega[n];
      n -= ilen;
    }

    if (pstyle == ISO) {
      ilen = 1;
      if (n < ilen) return omega_dot[n];
      n -= ilen;
    } else {
      ilen = 6;
      if (n < ilen) return omega_dot[n];
      n -= ilen;
    }
  }

  double volume;
  double kt = boltz * t_target;
  double lkt_press = kt;
  int ich;
  if (dimension == 3) volume = domain->xprd * domain->yprd * domain->zprd;
  else volume = domain->xprd * domain->yprd;

  if (tstat_flag) {
    ilen = mtchain;
    if (n < ilen) {
      ich = n;
      if (ich == 0)
        return ke_target * eta[0];
      else
        return kt * eta[ich];
    }
    n -= ilen;
    ilen = mtchain;
    if (n < ilen) {
      ich = n;
      if (ich == 0)
        return 0.5*eta_mass[0]*eta_dot[0]*eta_dot[0];
      else
        return 0.5*eta_mass[ich]*eta_dot[ich]*eta_dot[ich];
    }
    n -= ilen;
  }

  if (pstat_flag) {
    if (pstyle == ISO) {
      ilen = 1;
      if (n < ilen)
        return p_hydro*(volume) / nktv2p;
      n -= ilen;
    } else {
      ilen = 6;
      if (n < ilen) {
        if (n > 2) return 0.0;
        else if (p_flag[n])
          return p_hydro*(volume) / (pdim*nktv2p);
        else
          return 0.0;
      }
      n -= ilen;
    }

    if (pstyle == ISO) {
      ilen = 1;
      if (n < ilen)
        return pdim*0.5*omega_dot[n]*omega_dot[n]*omega_mass[n];
      n -= ilen;
    } else {
      ilen = 6;
      if (n < ilen) {
        if (p_flag[n])
          return 0.5*omega_dot[n]*omega_dot[n]*omega_mass[n];
        else return 0.0;
      }
      n -= ilen;
    }

  }

  return 0.0;
}

/* ---------------------------------------------------------------------- */

void FixMTTKNHC::reset_target(double t_new)
{
  t_target = t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

void FixMTTKNHC::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dthalf = 0.5 * update->dt;
  dt4 = 0.25 * update->dt;
  dt8 = 0.125 * update->dt;
  dto = dthalf;
}

/* ----------------------------------------------------------------------
   extract thermostat properties
------------------------------------------------------------------------- */

void *FixMTTKNHC::extract(const char *str, int &dim)
{
  dim=0;
  if (tstat_flag && strcmp(str,"t_target") == 0) {
    return &t_target;
  } else if (tstat_flag && strcmp(str,"t_start") == 0) {
    return &t_start;
  } else if (tstat_flag && strcmp(str,"t_stop") == 0) {
    return &t_stop;
  } else if (tstat_flag && strcmp(str,"mtchain") == 0) {
    return &mtchain;
  }
  dim=1;
  if (tstat_flag && strcmp(str,"eta") == 0) {
    return &eta;
  } else if (pstat_flag && strcmp(str,"p_flag") == 0) {
    return &p_flag;
  } else if (pstat_flag && strcmp(str,"p_start") == 0) {
    return &p_start;
  } else if (pstat_flag && strcmp(str,"p_stop") == 0) {
    return &p_stop;
  } else if (pstat_flag && strcmp(str,"p_target") == 0) {
    return &p_target;
  }
  return NULL;
}

/* ----------------------------------------------------------------------
   perform half-step update of chain thermostat variables
------------------------------------------------------------------------- */

void FixMTTKNHC::nhc_temp_integrate()
{
  int ich;
  double expfac;
  double kecurrent = tdof * boltz * t_current;
  double G1_add = -ndof_baro*boltz*t_target;
  for (int i=0;i<3;i++) G1_add += omega_mass[i]*omega_dot[i]*omega_dot[i];
  for (int i=3;i<6;i++) G1_add += omega_mass[i]*omega_dot[i]*omega_dot[i];

  if (eta_mass[0] > 0.0)
    eta_dotdot[0] = (kecurrent - ke_target + G1_add)/eta_mass[0];
  else eta_dotdot[0] = 0.0;

  for (ich = mtchain-1; ich > 0; ich--) eta_dotdot[ich] = (eta_mass[ich-1]*eta_dot[ich-1]*eta_dot[ich-1]-boltz*t_target)/eta_mass[ich];

  for (ich = mtchain-1; ich > 0; ich--) {
    expfac = exp(-dt8*eta_dot[ich+1]);
    eta_dot[ich] *= expfac;
    eta_dot[ich] += eta_dotdot[ich] * dt4;
    eta_dot[ich] *= expfac;
  }

  expfac = exp(-dt8*eta_dot[1]);
  eta_dot[0] *= expfac;
  eta_dot[0] += eta_dotdot[0] * dt4;
  eta_dot[0] *= expfac;

  factor_eta = exp(-dthalf*eta_dot[0]);
  nh_v_temp();

  // rescale temperature due to velocity scaling
  // should not be necessary to explicitly recompute the temperature

  t_current *= factor_eta*factor_eta;
  kecurrent = tdof * boltz * t_current;

  for (ich = 0; ich < mtchain; ich++)
    eta[ich] += dthalf*eta_dot[ich];

  if (eta_mass[0] > 0.0)
    eta_dotdot[0] = (kecurrent - ke_target + G1_add)/eta_mass[0];
  else eta_dotdot[0] = 0.0;

  eta_dot[0] *= expfac;
  eta_dot[0] += eta_dotdot[0] * dt4;
  eta_dot[0] *= expfac;

  for (ich = 1; ich < mtchain; ich++) {
    eta_dotdot[ich] = (eta_mass[ich-1]*eta_dot[ich-1]*eta_dot[ich-1]-boltz*t_target)/eta_mass[ich];
    expfac = exp(-dt8*eta_dot[ich+1]);
    eta_dot[ich] *= expfac;
    eta_dot[ich] += eta_dotdot[ich] * dt4;
    eta_dot[ich] *= expfac;
  }
#ifdef DEBUG
printf("%-30s", "DEBUG T CHAIN TVEL");
for (int i=0;i<3;i++) printf("%20.12f",eta_dot[i]);
printf("\n");
#endif
}


/* ----------------------------------------------------------------------
   perform half-step barostat scaling of velocities
-----------------------------------------------------------------------*/

void FixMTTKNHC::nh_v_press()
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;


  // omega is not used, except for book-keeping

  for (int i = 0; i < 6; i++) omega[i] += dto*omega_dot[i];

  // convert pertinent atoms and rigid bodies to lamda coords

  if (allremap) domain->x2lamda(nlocal);
  else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & dilate_group_bit)
        domain->x2lamda(x[i],x[i]);
  }

  if (nrigid)
    for (int i = 0; i < nrigid; i++)
      modify->fix[rfix[i]]->deform(0);

  double rotmat[6];
  double tmp,ump;
  // Compute the rotation matrix
  rotmat[0] = exp(omega_dot[0]*dto);
  rotmat[1] = exp(omega_dot[1]*dto);
  rotmat[2] = exp(omega_dot[2]*dto);
  // xy
  tmp = 0.5*dto*(omega_dot[0]-omega_dot[1]);
  tmp *= tmp;
  rotmat[3] = omega_dot[3]*dto* exp( 0.5*dto*(omega_dot[0]+omega_dot[1]))* (  (1.0 + A2*tmp*(1.0 + A4*tmp*(1.0 + A6*tmp*(1.0 + A8*tmp*(1.0+A10*tmp))))));
  // xz
  tmp = 0.5*dto*(omega_dot[2]-omega_dot[0]);
  tmp *= tmp;
  rotmat[4] = omega_dot[4]*dto* exp( 0.5*dto*(omega_dot[2]+omega_dot[0]))* (  (1.0 + A2*tmp*(1.0 + A4*tmp*(1.0 + A6*tmp*(1.0 + A8*tmp*(1.0+A10*tmp))))));
  tmp = 0.5*dto*(omega_dot[0]-omega_dot[1]);
  ump = 0.5*dto*(omega_dot[1]-omega_dot[2]);
  rotmat[4] += omega_dot[3]*omega_dot[5]*dto*dto* exp( (omega_dot[0]+omega_dot[1]+omega_dot[2])/3.0*dto) *
    (0.5+B0*(tmp*tmp+tmp*ump+ump*ump)+B1*tmp*tmp*tmp+B2*tmp*tmp*ump-B2*tmp*ump*ump-B1*ump*ump*ump
    +B3*tmp*tmp*tmp*tmp + B4*tmp*tmp*tmp*ump + B5*tmp*tmp*ump*ump + B4*tmp*ump*ump*ump + B5*ump*ump*ump*ump  );
  // yz
  tmp = 0.5*dto*(omega_dot[1]-omega_dot[2]);
  tmp *= tmp;
  rotmat[5] = omega_dot[5]*dto* exp( 0.5*dto*(omega_dot[1]+omega_dot[2]))* (  (1.0 + A2*tmp*(1.0 + A4*tmp*(1.0 + A6*tmp*(1.0 + A8*tmp*(1.0+A10*tmp))))));
  #ifdef DEBUG
  printf("%-30s", "DEBUG B UPDATE ROTMAT");
  for (int i = 0; i < 6; i++) printf("%20.12f", rotmat[i]);
  printf("\n");
  #endif


  // Update the cell, positions in fractional coordinates are automatically updated
  // Order of updates matters!
  domain->xz = rotmat[0]*domain->xz + rotmat[3]*domain->yz+rotmat[4]*domain->boxhi[2];
  domain->xy = rotmat[0]*domain->xy + rotmat[3]*domain->boxhi[1];
  domain->yz = rotmat[1]*domain->yz + rotmat[5]*domain->boxhi[2];
  domain->boxhi[0] = rotmat[0]*domain->boxhi[0];
  domain->boxhi[1] = rotmat[1]*domain->boxhi[1];
  domain->boxhi[2] = rotmat[2]*domain->boxhi[2];

  domain->set_global_box();
  domain->set_local_box();

  // convert pertinent atoms and rigid bodies back to box coords
  if (allremap) domain->lamda2x(nlocal);
  else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & dilate_group_bit)
        domain->lamda2x(x[i],x[i]);
  }


  #ifdef DEBUG
  printf("%-30s", "DEBUG B UPDATE CELL");
  printf("%20.12f%20.12f%20.12f%20.12f%20.12f%20.12f\n",
    domain->boxhi[0]-domain->boxlo[0],domain->boxhi[1]-domain->boxlo[1],domain->boxhi[2]-domain->boxlo[2],domain->xy,domain->xz,domain->yz);
  #endif

  // Compute another rotation matrix, exp(-A*t)
  rotmat[0] = exp(-omega_dot[0]*dto);
  rotmat[1] = exp(-omega_dot[1]*dto);
  rotmat[2] = exp(-omega_dot[2]*dto);
  // xy
  tmp = -0.5*dto*(omega_dot[0]-omega_dot[1]);
  tmp *= tmp;
  rotmat[3] = -omega_dot[3]*dto* exp( -0.5*dto*(omega_dot[0]+omega_dot[1]))* (  (1.0 + A2*tmp*(1.0 + A4*tmp*(1.0 + A6*tmp*(1.0 + A8*tmp*(1.0+A10*tmp))))));
  // xz
  tmp = -0.5*dto*(omega_dot[2]-omega_dot[0]);
  tmp *= tmp;
  rotmat[4] = -omega_dot[4]*dto* exp( -0.5*dto*(omega_dot[2]+omega_dot[0]))* (  (1.0 + A2*tmp*(1.0 + A4*tmp*(1.0 + A6*tmp*(1.0 + A8*tmp*(1.0+A10*tmp))))));
  tmp = -0.5*dto*(omega_dot[0]-omega_dot[1]);
  ump = -0.5*dto*(omega_dot[1]-omega_dot[2]);
  rotmat[4] += omega_dot[3]*omega_dot[5]*dto*dto* exp(- (omega_dot[0]+omega_dot[1]+omega_dot[2])/3.0*dto) *
    (0.5+B0*(tmp*tmp+tmp*ump+ump*ump)+B1*tmp*tmp*tmp+B2*tmp*tmp*ump-B2*tmp*ump*ump-B1*ump*ump*ump
    +B3*tmp*tmp*tmp*tmp + B4*tmp*tmp*tmp*ump + B5*tmp*tmp*ump*ump + B4*tmp*ump*ump*ump + B5*ump*ump*ump*ump  );
  // yz
  tmp = -0.5*dto*(omega_dot[1]-omega_dot[2]);
  tmp *= tmp;
  rotmat[5] = -omega_dot[5]*dto* exp( -0.5*dto*(omega_dot[1]+omega_dot[2]))* (  (1.0 + A2*tmp*(1.0 + A4*tmp*(1.0 + A6*tmp*(1.0 + A8*tmp*(1.0+A10*tmp))))));
  // Scale all
  if (!vol_constraint) {
      double fac = exp( -dto*(omega_dot[0]+omega_dot[1]+omega_dot[2])/tdof);
      rotmat[0] *= fac;
      rotmat[1] *= fac;
      rotmat[2] *= fac;
      rotmat[3] *= fac;
      rotmat[4] *= fac;
      rotmat[5] *= fac;
  }

  #ifdef DEBUG
  printf("%-30s", "DEBUG B UPDATE ROTMAT");
  for (int i = 0; i < 6; i++) printf("%20.12f", rotmat[i]);
  printf("\n");
  #endif

  // Update velocities
  if (which == NOBIAS) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        v[i][0] = rotmat[0]*v[i][0] + rotmat[3]*v[i][1] + rotmat[4]*v[i][2];
        v[i][1] = rotmat[1]*v[i][1] + rotmat[5]*v[i][2];
        v[i][2] = rotmat[2]*v[i][2];
      }
    }
  } else if (which == BIAS) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        temperature->remove_bias(i,v[i]);
        v[i][0] = rotmat[0]*v[i][0] + rotmat[3]*v[i][1] + rotmat[4]*v[i][2];
        v[i][1] = rotmat[1]*v[i][1] + rotmat[5]*v[i][2];
        v[i][2] = rotmat[2]*v[i][2];
        temperature->restore_bias(i,v[i]);
      }
    }
  }

  #ifdef DEBUG
  for (int i = 0; i < 2; i++) {
    if (mask[i] & groupbit) {
      printf("%-24s %5d%20.12f%20.12f%20.12f\n", "DEBUG B UPDATE POS", i, x[i][0], x[i][1], x[i][2]);
    }
  }
  for (int i = 0; i < 2; i++) {
    if (mask[i] & groupbit) {
      printf("%-24s %5d%20.12f%20.12f%20.12f\n", "DEBUG B UPDATE VEL", i, v[i][0], v[i][1], v[i][2]);
    }
  }
  #endif
}

/* ----------------------------------------------------------------------
   perform half-step update of velocities
-----------------------------------------------------------------------*/

void FixMTTKNHC::nve_v()
{
  double dtfm;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  if (rmass) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        v[i][0] += dtfm*f[i][0];
        v[i][1] += dtfm*f[i][1];
        v[i][2] += dtfm*f[i][2];
      }
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm*f[i][0];
        v[i][1] += dtfm*f[i][1];
        v[i][2] += dtfm*f[i][2];
      }
    }
  }
  #ifdef DEBUG
  for (int i = 0; i < 2; i++) {
    if (mask[i] & groupbit) {
      printf("%-24s %5d%20.12f%20.12f%20.12f\n", "DEBUG V UPDATE VEL", i, v[i][0], v[i][1], v[i][2]);
    }
  }
  #endif

}

/* ----------------------------------------------------------------------
   perform full-step update of positions
-----------------------------------------------------------------------*/

void FixMTTKNHC::nve_x()
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // x update by full step only for atoms in group
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];
    }
  }
  #ifdef DEBUG
  for (int i = 0; i < 2; i++) {
    if (mask[i] & groupbit) {
      printf("%-24s %5d%20.12f%20.12f%20.12f\n", "DEBUG V UPDATE POS", i, x[i][0], x[i][1], x[i][2]);
    }
  }
  #endif
}

/* ----------------------------------------------------------------------
   perform half-step thermostat scaling of velocities
-----------------------------------------------------------------------*/

void FixMTTKNHC::nh_v_temp()
{
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (which == NOBIAS) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        v[i][0] *= factor_eta;
        v[i][1] *= factor_eta;
        v[i][2] *= factor_eta;
      }
    }
  } else if (which == BIAS) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        temperature->remove_bias(i,v[i]);
        v[i][0] *= factor_eta;
        v[i][1] *= factor_eta;
        v[i][2] *= factor_eta;
        temperature->restore_bias(i,v[i]);
      }
    }
  }
  #ifdef DEBUG
  for (int i = 0; i < 2; i++) {
    if (mask[i] & groupbit) {
      printf("%-24s %5d%20.12f%20.12f%20.12f\n", "DEBUG T UPDATE VEL", i, v[i][0], v[i][1], v[i][2]);
    }
  }
  #endif
}


/* ----------------------------------------------------------------------
   compute target temperature and kinetic energy
-----------------------------------------------------------------------*/

void FixMTTKNHC::compute_temp_target()
{
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  t_target = t_start + delta * (t_stop-t_start);
  ke_target = tdof * boltz * t_target;
}

/* ----------------------------------------------------------------------
   compute hydrostatic target pressure
-----------------------------------------------------------------------*/

void FixMTTKNHC::compute_press_target()
{
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  p_hydro = 0.0;
  for (int i = 0; i < 3; i++)
    if (p_flag[i]) {
      p_target[i] = p_start[i] + delta * (p_stop[i]-p_start[i]);
      p_hydro += p_target[i];
    }
  if (pdim > 0) p_hydro /= pdim;

  if (pstyle == TRICLINIC)
    for (int i = 3; i < 6; i++)
      p_target[i] = p_start[i] + delta * (p_stop[i]-p_start[i]);

}

/* ----------------------------------------------------------------------
   update omega_dot, omega
-----------------------------------------------------------------------*/

void FixMTTKNHC::nh_omega_dot()
{
  double f_diag[3];
  double f_trace,f_omega,volume;

  if (dimension == 3) volume = domain->xprd*domain->yprd*domain->zprd;
  else volume = domain->xprd*domain->yprd;

  #ifdef DEBUG
  printf("%-30s%20.12f\n", "DEBUG B CHAINVEL", eta_dot[0]);
  printf("%-30s", "DEBUG B UPDATE GTENS");
  for (int i = 0; i < 3; i++) printf("%20.12f", ((p_current[i]-p_hydro)*volume/nktv2p+boltz*t_current)/omega_mass[i]);
  for (int i = 3; i < 6; i++) printf("%20.12f", ((p_current[i])*volume/nktv2p)/omega_mass[i]);
  printf("\n");
  #endif

  // iL v_{xi} v_g h/8
  double expfac = exp(-dt8*eta_dot[0]);
  for (int i=0; i<6; i++) {
    omega_dot[i] *= expfac;
  }

  // iL G_g h/4
  for (int i=0; i<3; i++)
      f_diag[i] = ((p_current[i]-p_hydro)*volume/nktv2p+boltz*t_current)/omega_mass[i]*dt4;
  if (vol_constraint) {
    f_trace = (f_diag[0] + f_diag[1] + f_diag[2])/3.0;
    for (int i=0; i<3; i++)
      f_diag[i] -= f_trace;
  }
  for (int i=0; i<3; i++) {
      omega_dot[i] += f_diag[i];
  }

  if (pstyle==TRICLINIC) {
    // Mind the Voigt notation!
    omega_dot[3] += p_current[5]*volume/omega_mass[3]*dt4/nktv2p;
    omega_dot[4] += p_current[4]*volume/omega_mass[4]*dt4/nktv2p;
    omega_dot[5] += p_current[3]*volume/omega_mass[5]*dt4/nktv2p;
  }

  for (int i=0; i<6; i++) {
    omega_dot[i] *= expfac;
  }
  #ifdef DEBUG
  printf("%-30s", "DEBUG B UPDATE BVEL");
  for (int i = 0; i < 6; i++) printf("%20.12f", omega_dot[i]);
  printf("\n");
  #endif
}

/* ----------------------------------------------------------------------
  if any tilt ratios exceed limits, set flip = 1 and compute new tilt values
  do not flip in x or y if non-periodic (can tilt but not flip)
    this is b/c the box length would be changed (dramatically) by flip
  if yz tilt exceeded, adjust C vector by one B vector
  if xz tilt exceeded, adjust C vector by one A vector
  if xy tilt exceeded, adjust B vector by one A vector
  check yz first since it may change xz, then xz check comes after
  if any flip occurs, create new box in domain
  image_flip() adjusts image flags due to box shape change induced by flip
  remap() puts atoms outside the new box back into the new box
  perform irregular on atoms in lamda coords to migrate atoms to new procs
  important that image_flip comes before remap, since remap may change
    image flags to new values, making eqs in doc of Domain:image_flip incorrect
------------------------------------------------------------------------- */

void FixMTTKNHC::pre_exchange()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;

  // flip is only triggered when tilt exceeds 0.5 by DELTAFLIP
  // this avoids immediate re-flipping due to tilt oscillations

  double xtiltmax = (0.5+DELTAFLIP)*xprd;
  double ytiltmax = (0.5+DELTAFLIP)*yprd;

  int flipxy,flipxz,flipyz;
  flipxy = flipxz = flipyz = 0;

  if (domain->yperiodic) {
    if (domain->yz < -ytiltmax) {
      domain->yz += yprd;
      domain->xz += domain->xy;
      flipyz = 1;
    } else if (domain->yz >= ytiltmax) {
      domain->yz -= yprd;
      domain->xz -= domain->xy;
      flipyz = -1;
    }
  }

  if (domain->xperiodic) {
    if (domain->xz < -xtiltmax) {
      domain->xz += xprd;
      flipxz = 1;
    } else if (domain->xz >= xtiltmax) {
      domain->xz -= xprd;
      flipxz = -1;
    }
    if (domain->xy < -xtiltmax) {
      domain->xy += xprd;
      flipxy = 1;
    } else if (domain->xy >= xtiltmax) {
      domain->xy -= xprd;
      flipxy = -1;
    }
  }

  int flip = 0;
  if (flipxy || flipxz || flipyz) flip = 1;

  if (flip) {
    domain->set_global_box();
    domain->set_local_box();

    domain->image_flip(flipxy,flipxz,flipyz);

    double **x = atom->x;
    imageint *image = atom->image;
    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++) domain->remap(x[i],image[i]);

    domain->x2lamda(atom->nlocal);
    irregular->migrate_atoms();
    domain->lamda2x(atom->nlocal);
  }
}

/* ----------------------------------------------------------------------
   memory usage of Irregular
------------------------------------------------------------------------- */

double FixMTTKNHC::memory_usage()
{
  double bytes = 0.0;
  if (irregular) bytes += irregular->memory_usage();
  return bytes;
}
