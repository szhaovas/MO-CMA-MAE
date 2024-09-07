import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Tuple
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.utils.pareto_front import compute_masked_pareto_front
from qdax.types import Fitness, Genotype, Descriptor
from ribs._utils import (
    check_batch_shape,
    check_solution_batch_dim,
    check_finite,
)

from src.moribs.archives import PFCVTArchive
from ._pf_utils import (
    batch_entry_pf,
    compute_moqd_score,
    compute_best_index,
    compute_total_numvisits,
)


class NSGA2Repertoire(GARepertoire):
    """NSGA2Repertoire but also stores measures/behavior descriptors
    Modified from:
        <https://github.com/adaptive-intelligent-robotics/MOME_PGX/blob/main/qdax/core/containers/nsga2_repertoire.py>
    """

    descriptors: Descriptor

    @jax.jit
    def _compute_crowding_distances(
        self, fitnesses: Fitness, mask: jnp.ndarray
    ) -> jnp.ndarray:
        # Retrieve only non masked solutions
        num_solutions = fitnesses.shape[0]
        num_objective = fitnesses.shape[1]
        if num_solutions <= 2:
            return jnp.array([jnp.inf] * num_solutions)

        else:
            # Sort solutions on each objective
            mask_dist = jnp.column_stack([mask] * fitnesses.shape[1])
            score_amplitude = jnp.max(fitnesses, axis=0) - jnp.min(
                fitnesses, axis=0
            )
            dist_fitnesses = (
                fitnesses
                + 3 * score_amplitude * jnp.ones_like(fitnesses) * mask_dist
            )
            sorted_index = jnp.argsort(dist_fitnesses, axis=0)
            srt_fitnesses = fitnesses[sorted_index, jnp.arange(num_objective)]

            # Calculate the norm for each objective - set to NaN if all values are equal
            norm = jnp.max(srt_fitnesses, axis=0) - jnp.min(
                srt_fitnesses, axis=0
            )

            # get the distances
            dists = jnp.vstack(
                [srt_fitnesses, jnp.full(num_objective, jnp.inf)]
            ) - jnp.vstack(
                [jnp.full(num_objective, -jnp.inf), srt_fitnesses]
            )

            # Prepare the distance to last and next vectors
            dist_to_last, dist_to_next = dists, dists
            dist_to_last = dists[:-1] / norm
            dist_to_next = dists[1:] / norm

            # Sum up the distances and reorder
            j = jnp.argsort(sorted_index, axis=0)
            crowding_distances = (
                jnp.sum(
                    (
                        dist_to_last[j, jnp.arange(num_objective)]
                        + dist_to_next[j, jnp.arange(num_objective)]
                    ),
                    axis=1,
                )
                / num_objective
            )

            return crowding_distances

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_fitnesses: Fitness,
        batch_of_descriptors: Descriptor,
    ):
        # Initialnumber of solutions:
        # original_population_size = self.size

        # All the candidates
        candidates = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            self.genotypes,
            batch_of_genotypes,
        )

        candidate_fitnesses = jnp.concatenate(
            (self.fitnesses, batch_of_fitnesses)
        )
        candidate_descriptors = jnp.concatenate(
            (self.descriptors, batch_of_descriptors)
        )

        first_leaf = jax.tree_util.tree_leaves(candidates)[0]
        num_candidates = first_leaf.shape[0]

        def compute_current_front(
            val: Tuple[jnp.ndarray, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            to_keep_index, _ = val

            # mask the individual that are already kept
            front_index = compute_masked_pareto_front(
                candidate_fitnesses, mask=to_keep_index
            )

            # Add new indexes
            to_keep_index = to_keep_index + front_index

            # Update front & number of solutions
            return to_keep_index, front_index

        def condition_fn_1(val: Tuple[jnp.ndarray, jnp.ndarray]) -> bool:
            to_keep_index, _ = val
            return sum(to_keep_index) < self.size  # type: ignore

        # get indexes of all first successive fronts and indexes of the last front
        to_keep_index, front_index = jax.lax.while_loop(
            condition_fn_1,
            compute_current_front,
            (
                jnp.zeros(num_candidates, dtype=bool),
                jnp.zeros(num_candidates, dtype=bool),
            ),
        )

        # remove the indexes of the last front - gives first indexes to keep
        new_index = (
            jnp.arange(start=1, stop=len(to_keep_index) + 1) * to_keep_index
        )
        new_index = new_index * (~front_index)
        to_keep_index = new_index > 0

        # Compute crowding distances
        crowding_distances = self._compute_crowding_distances(
            candidate_fitnesses, ~front_index
        )
        crowding_distances = crowding_distances * (front_index)
        highest_dist = jnp.argsort(crowding_distances)

        def add_to_front(
            val: Tuple[jnp.ndarray, float]
        ) -> Tuple[jnp.ndarray, Any]:
            front_index, num = val
            front_index = front_index.at[highest_dist[-num]].set(True)
            num = num + 1
            val = front_index, num
            return val

        def condition_fn_2(val: Tuple[jnp.ndarray, jnp.ndarray]) -> bool:
            front_index, _ = val
            return sum(to_keep_index + front_index) < self.size  # type: ignore

        # add the individuals with the highest distances
        front_index, _num = jax.lax.while_loop(
            condition_fn_2,
            add_to_front,
            (jnp.zeros(num_candidates, dtype=bool), 0),
        )

        # update index
        to_keep_index = to_keep_index + front_index

        # go from boolean vector to indices - offset by 1
        indices = jnp.arange(start=1, stop=num_candidates + 1) * to_keep_index

        # get rid of the zeros (that correspond to the False from the mask)
        fake_indice = num_candidates + 1  # bigger than all the other indices
        indices = jnp.where(indices == 0, fake_indice, indices)

        # sort the indices to remove the fake indices
        indices = jnp.sort(indices)[: self.size]

        # remove the offset
        indices = indices - 1

        # keep only the survivors
        new_candidates = jax.tree_util.tree_map(
            lambda x: x[indices], candidates
        )
        new_scores = candidate_fitnesses[indices]
        new_descriptors = candidate_descriptors[indices]

        new_repertoire = self.replace(
            genotypes=new_candidates,
            fitnesses=new_scores,
            descriptors=new_descriptors,
        )

        # added_list = to_keep_index[original_population_size:]
        # removed_count = original_population_size - jnp.sum(to_keep_index[:original_population_size])

        return new_repertoire  # [added_list, removed_count] # type: ignore

    @classmethod
    def init(  # type: ignore
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        population_size: int,
    ) -> GARepertoire:
        # create default fitnesses
        default_fitnesses = -jnp.inf * jnp.ones(
            shape=(population_size, fitnesses.shape[-1])
        )

        # create default genotypes
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(shape=(population_size,) + x.shape[1:]),
            genotypes,
        )

        default_descriptors = jnp.zeros(
            shape=(population_size, descriptors.shape[-1])
        )

        # create an initial repertoire with those default values
        repertoire = cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
        )

        new_repertoire = repertoire.add(genotypes, fitnesses, descriptors)

        return new_repertoire  # type: ignore


class NSGA2Archive(PFCVTArchive):
    def __init__(
        self,
        *,
        solution_dim,
        objective_dim,
        reference_point,
        cells,
        ranges,
        pop_size,
        seed=None,
        samples=100_000,
    ):
        
        PFCVTArchive.__init__(
            self,
            solution_dim=solution_dim,
            objective_dim=objective_dim,
            reference_point=reference_point,
            cells=cells,
            ranges=ranges,
            bias_sampling=False,
            init_discount=1,
            alpha=1,
            max_pf_size=None,
            hvi_cutoff_threshold=None,
            seed=seed,
            samples=samples,
        )

        self._pop_size = pop_size

        self.main = NSGA2Repertoire.init(
            # Workaround for NSGA2Repertoire needing an init_population.
            genotypes=jnp.zeros((1, solution_dim)),
            fitnesses=jnp.full((1, objective_dim), -jnp.inf),
            descriptors=jnp.full((1, self.measure_dim), 0),
            population_size=self.pop_size,
        )

        self.jax_rng_key = jax.random.key(seed)

    @property
    def pop_size(self):
        return self._pop_size

    def qd_update(self, **fields):
        sols = np.array(self.main.genotypes, dtype=np.float64)
        objs = np.array(self.main.fitnesses, dtype=np.float64)
        meas = np.array(self.main.descriptors, dtype=np.float64)
        filled_indices = np.where(np.all(objs != -np.inf, axis=-1))

        data = {
            "solution": sols[filled_indices],
            "objective": objs[filled_indices],
            "measures": meas[filled_indices],
            **fields,
        }

        add_info = self._store.add(
            self.index_of(data["measures"]),
            data,
            {
                "hvi_cutoff_threshold": self.hvi_cutoff_threshold,
                # The ArchiveBase class maintains "_objective_sum" when calculating
                # sum, so we use self._objective_sum here to stay compatible.
                "hypervolume_sum": self._objective_sum,
                "total_numvisits": self.total_numvisits
            },
            [batch_entry_pf, compute_moqd_score, compute_best_index, compute_total_numvisits],
        )

        # Updates passive archive QD metrics.
        hypervolume_sum = add_info.pop("hypervolume_sum")
        best_index = add_info.pop("best_index")
        total_numvisits = add_info.pop("total_numvisits")
        if not np.all(add_info["status"] == 0):
            self._stats_update(hypervolume_sum, best_index, total_numvisits)

    def add_single(self, solution, objective, measures, **fields):
        raise NotImplementedError("Please use batch add() for NSGA2.")

    def add(
        self,
        solution: np.ndarray,
        objective: np.ndarray,
        measures: np.ndarray,
        **fields,
    ):
        """Inserts the new solutions and objectives into the NSGA2Population
        object.

        NOTE: Call ``qd_update()`` manually before accessing passive archive.

        Args:
            solution (np.ndarray): a 2D array containing
            ``batch_size * solution_dim`` entries.
            objective (np.ndarray): a 2D array containing
            ``batch_size * objective_dim`` entries.
            measures (np.ndarray): a 2D array containing
            ``batch_size * measure_dim`` entries.

        Returns:
            add_info (dict): Dummy add_info with all values and states set to 0.
        """
        batch_size = solution.shape[0]
        check_batch_shape(
            solution, "solution", self.solution_dim, "solution_dim", ""
        )
        check_solution_batch_dim(solution, "solution", batch_size, extra_msg="")

        check_batch_shape(
            objective, "objective", self.objective_dim, "objective_dim", ""
        )
        check_solution_batch_dim(
            objective, "objective", batch_size, extra_msg=""
        )

        check_batch_shape(
            measures, "measures", self.measure_dim, "measure_dim", ""
        )
        check_solution_batch_dim(measures, "measures", batch_size, extra_msg="")
        check_finite(measures, "measures")

        # Update the NSGA2 population to obtain individuals that should go into the passive archive.
        self.main = self.main.add(
            batch_of_genotypes=jnp.array(solution),
            batch_of_fitnesses=jnp.array(objective),
            batch_of_descriptors=jnp.array(measures),
        )

        # Dummy add_info.
        return {
            "status": np.zeros(batch_size, dtype=np.int32),
            "value": np.zeros(batch_size, dtype=np.float64),
        }

    @property
    def empty(self):
        """Since passive archive is not always in sync with the main pop,
        checks whether the main PF is empty."""
        return jnp.all(self.main.fitnesses == -jnp.inf)

    def sample_elites(self, n):
        """Since passive archive is not always in sync with the main pop,
        samples an elite from the main pop instead of passive archive
        """
        if self.empty:
            raise IndexError("No elements in archive.")

        samples, self.jax_rng_key = self.main.sample(self.jax_rng_key, n)

        return {"solution": samples}
