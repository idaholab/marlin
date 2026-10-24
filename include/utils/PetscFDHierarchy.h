/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/
#pragma once

#include "LinearFDOperator.h"

#include <petscdmda.h>

#include <array>
#include <cstdint>
#include <vector>

class DomainAction;

/**
 * PETSc pieces of the implicit finite-difference solver's multigrid hierarchy:
 * a fine DMDA on Marlin's real space partition, its coarsened levels, the
 * rediscretised coarse level operators and the cell-centred interpolation.
 *
 * Axis convention: Marlin tensors are row-major [n_0, n_1, n_2] with the last
 * axis contiguous, DMDA numbers x fastest. DMDA axis a is therefore Marlin axis
 * dim - 1 - a, which makes the owned block of a Marlin tensor exactly this
 * rank's part of a DMDA global vector, with no reordering.
 */
namespace PetscFD
{

/// Marlin's real space partition, in Marlin axis order
struct Partition
{
  unsigned int dim;
  /// global number of cells per axis
  std::array<int64_t, 3> n;
  /// number of ranks per axis and this rank's index per axis
  std::array<unsigned int, 3> parts, index;
  /// owned cells of every slab, per axis
  std::array<std::vector<PetscInt>, 3> counts;
  /// the domain's communicator
  MPI_Comm comm;
};

/// gather the partition from the domain (collective)
Partition gatherPartition(const DomainAction & domain);

/// communicator with the ranks renumbered in the DMDA's rank order (free with MPI_Comm_free)
MPI_Comm dmdaComm(const Partition & partition);

/// stencil width for the DMDA: max |offset|, at least 1
PetscInt stencilWidth(const std::vector<FDStencilEntry> & stencil);
bool needsBoxStencil(const std::vector<FDStencilEntry> & stencil);

/**
 * Number of multigrid levels (fine level included). A level is coarsened while
 * every global extent is even, the coarse extent stays >= 8 cells, and every rank
 * keeps at least stencil_width cells per axis (a DMDA requirement).
 */
unsigned int countLevels(const Partition & partition, PetscInt stencil_width);

/// multigrid hierarchy of DMDAs and interpolations
struct Hierarchy
{
  Hierarchy() = default;
  Hierarchy(const Hierarchy &) = delete;
  Hierarchy & operator=(const Hierarchy &) = delete;
  ~Hierarchy();

  /// DMDAs, fine level first (dms[0] is on Marlin's partition)
  std::vector<DM> dms;
  /// interpolations[l] maps level l + 1 (coarser) to level l
  std::vector<Mat> interpolations;
};

/**
 * Build the hierarchy on comm (from dmdaComm). The fine DMDA is periodic on
 * every axis and must own exactly this rank's Marlin block, which is checked.
 */
void buildHierarchy(const Partition & partition,
                    MPI_Comm comm,
                    unsigned int levels,
                    PetscInt stencil_width,
                    bool box,
                    Hierarchy & hierarchy);

/**
 * Assemble scale * (the stencil) on the periodic DMDA da into B (from
 * createStencilMatrix), replacing its values. Stencil offsets are in Marlin axis order,
 * and entries that land on the same column (same offset, or offsets that wrap
 * onto each other on a small periodic grid) are summed.
 */
void
assembleStencilOperator(DM da, const std::vector<FDStencilEntry> & stencil, PetscReal scale, Mat B);

/**
 * An AIJ matrix on the DMDA da (of the DM's matrix type) preallocated for exactly the
 * columns of the stencil, which assembleStencilOperator then fills. Any stencil with
 * the same offsets can be assembled into it.
 */
Mat createStencilMatrix(DM da, const std::vector<FDStencilEntry> & stencil);

/**
 * Cell-centred multilinear interpolation from the coarse to the fine DMDA (each
 * axis: 3/4 the coarse parent, 1/4 the coarse neighbour on the fine cell's side),
 * assembled as AIJ. Columns are global coarse indices from the coarse ownership
 * ranges, since DMCoarsen chooses the coarse partition.
 */
Mat cellCentredInterpolation(DM coarse, DM fine);

} // namespace PetscFD
