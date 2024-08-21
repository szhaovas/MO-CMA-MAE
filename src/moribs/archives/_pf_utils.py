import copy
import logging
import numpy as np

from itertools import combinations

from ribs.archives._add_status import AddStatus
import ribs.visualize


logger = logging.getLogger(__name__)


def weakly_dominates(this_objs, other_objs):
    return np.all(this_objs >= other_objs) and np.any(this_objs > other_objs)


def compute_crowding_distances(objective, boundary_inf):
    """Computes crowding distances as described in NSGA-II (see
    <https://ieeexplore-ieee-org.libproxy2.usc.edu/abstract/document/996017/>)

    For each objective dimension, sort the solutions, and assign crowding distances for this
    objective to be ``(objective[i+1] - objective[i-1]) / max(objective) - min(objective)``
    for each solution at sorted index ``i``.
    Solutions with max or min values along this objective are given crowding distance ``inf``.
    The final crowding distance for a solution is the sum of all its crowding distances along
    every objective.


    Args:
        objective (np.ndarray): Objective values. Should be a 2D array with
            dimension ``num_of_solutions * objective_dim``. ``num_of_solutions``
            must be at least 2.

    Returns:
        crowding_distances (np.ndarray): A 1D array of length ``num_of_solutions``
            containing the crowding distances for each solution.
    """
    assert np.all(objective >= 0)

    num_of_solutions, objective_dim = objective.shape
    if num_of_solutions < 2:
        raise ValueError(
            "compute_crowding_distances expects input objectives to have at least 2 rows,"
            f"but received objective with {num_of_solutions} rows."
        )

    crowding_distances = np.zeros((num_of_solutions,))
    for obj_idx in range(objective_dim):
        sort_indices = np.argsort(objective[:, obj_idx])
        norm_factor = (
            objective[sort_indices[-1], obj_idx] - objective[sort_indices[0], obj_idx]
        )

        # Other solutions are assigned normalized difference between nearest neighbors.
        for sol_idx in range(1, num_of_solutions - 1):
            crowding_distances[sort_indices[sol_idx]] += (
                objective[sort_indices[sol_idx + 1], obj_idx]
                - objective[sort_indices[sol_idx - 1], obj_idx]
            ) / norm_factor

        # When boundary_inf = True, boundary solutions are assigned inf crowding distance.
        if boundary_inf:
            crowding_distances[sort_indices[0]] = np.inf
            crowding_distances[sort_indices[-1]] = np.inf
        # When boundary_inf = False, boundary solutions are assigned one-sided distance to NN
        else:
            crowding_distances[sort_indices[0]] += (
                objective[sort_indices[1], obj_idx]
                - objective[sort_indices[0], obj_idx]
            ) / norm_factor
            crowding_distances[sort_indices[-1]] += (
                objective[sort_indices[-1], obj_idx]
                - objective[sort_indices[-2], obj_idx]
            ) / norm_factor

    return crowding_distances / objective_dim


def find_dominator(akv_indices, objectives):
    """Returns a dict containing for each solution its dominator indices.

    Checks for weak dominance under maximizing convention. Only solutions
    assigned to the same archive cell are compared. For global comparison,
    simply pass in all equal akv_indices.

    Args:
        akv_indices (array-like): Array of akv_indices at which new_data should be
            inserted.
        objectives (array-like): Objectives for the given akv_indices. Should be a 2D
            array with dimension ``num_of_solutions * objective_dim``. First dimension
            must have matching indices with ``akv_indices``.

    Returns:
        dominators (dict):
            - ``keys``: The indices from input solutions, i.e. range(len(akv_indices)).
            - ``values``: For each index, a list containing indices of other input
                    solutions which strictly dominate it.
    """
    batch_size = len(akv_indices)

    dominators = dict([(i, []) for i in range(batch_size)])
    for (i, this_obj), (j, other_obj) in combinations(enumerate(objectives), 2):
        # Only compare dominance when `this` and `other` are assigned to the same archive cell.
        if akv_indices[i] == akv_indices[j]:
            # `this` dominates `other` when:
            # 1. All `this` objs are at least as large as corresponding `other` objs.
            # 2. At least one `this` obj is larger than its corresponding `other` obj.
            if weakly_dominates(this_obj, other_obj):
                dominators[j].append(i)
            elif weakly_dominates(other_obj, this_obj):
                dominators[i].append(j)

    return dominators


def find_dominatee(akv_indices, objectives):
    """The mirror of _find_dominator. For each index, returns a list containing indices
    of other input solutions which it strictly dominates.
    """
    batch_size = len(akv_indices)

    dominatees = dict([(i, []) for i in range(batch_size)])
    for (i, this_obj), (j, other_obj) in combinations(enumerate(objectives), 2):
        if akv_indices[i] == akv_indices[j]:
            if weakly_dominates(this_obj, other_obj):
                dominatees[i].append(j)
            elif weakly_dominates(other_obj, this_obj):
                dominatees[j].append(i)

    return dominatees


def batch_entry_pf(indices, new_data, add_info, extra_args, occupied, cur_data):
    """The MO-CMA-ME counterpart to batch_entries_with_threshold (see
    <https://github.com/icaros-usc/pyribs/blob/master/ribs/archives/_transforms.py#L125>).

    Must be the first among transform functions passed to ArrayStore.add(),
    because it prepares downstream new_data for the following transforms.

    Args:
        indices (array-like): Array of indices at which new_data should be
            inserted.
        new_data (dict): New data for the given indices. Maps from field
            name to the array of new data for that field. Must have matching
            indices with ``indices``.
        add_info (dict): Information to return to the user about the
            addition process. Example info includes whether each entry was
            ultimately inserted into the store, as well as general statistics.
            For the first transform, this will be an empty dict.
        extra_args (dict): Additional arguments for the transform.
        occupied (array-like): Whether the given indices are currently
            occupied. Same as that given by :meth:`retrieve`. Not actually used
            since every archive cell is occupied by at least
        cur_data (dict): Data at the current indices in the store. Same as
            that given by :meth:`retrieve`. Must have matching indices with
            ``indices``.
    """
    # pylint: disable = unused-argument
    batch_size = len(indices)

    hvi_cutoff_threshold = extra_args["hvi_cutoff_threshold"]
    add_info["status"] = np.zeros(batch_size, dtype=np.int32)
    add_info["value"] = np.zeros(batch_size, dtype=np.float64)

    if np.any((new_data["objective"] > 100) | (new_data["objective"] < 0)):
        logger.error(
            f"Some objectives exceeded the [0, 100] range!\n"
            "\t objectives:\n"
            f"\t{new_data['objective']}".replace("\n", "\n\t")
        )

    for i, (multi_obj, pf, ocpd) in enumerate(zip(new_data["objective"], cur_data["pf"], occupied)):
        # Cannot add to PF here because adding here modifies PF for later
        #   solutions and creates bias.
        if ocpd:
            value, status = pf.hypervolume_improvement(-multi_obj)
        else:
            value, status = abs(np.prod(multi_obj)), 2

        # If a solution has IMPROVE_EXISTING add status, and
        # its hypervolume improvement value < hvi_cutoff_threshold,
        # its UHVI value and add status are set to 0.
        if (
            (not hvi_cutoff_threshold is None)
            and (status == AddStatus.IMPROVE_EXISTING)
            and (value < hvi_cutoff_threshold)
        ):
            value, status = 0, AddStatus.NOT_ADDED

        add_info["value"][i], add_info["status"][i] = value, status

    is_new = add_info["status"] == 2
    improve_existing = add_info["status"] == 1
    # Even if a solution passed the previous add check, it may still
    # have been dominated by another solution within batch assigned
    # to the same archive cell.
    dominators_within_batch = find_dominator(indices, new_data["objective"])
    nondominated_within_batch = [
        True if len(dom) == 0 else False for dom in dominators_within_batch.values()
    ]
    can_insert = (is_new | improve_existing) & nondominated_within_batch

    logger.info(
        f"Among {batch_size} generated solutions, {sum(is_new)} discover new cells, and {sum(improve_existing)} improve existing cells.\n"
        f"\t{sum(~np.array(nondominated_within_batch))} are dominated by other generated solutions within same archive cells.\n"
        f"\tOverall, {sum(can_insert)} solutions passed the add test and will be added."
    )

    # Return early if we cannot insert anything -- continuing would actually
    # throw a ValueError in aggregate() since index[can_insert] would be empty.
    if not np.any(can_insert):
        return np.array([], dtype=np.int32), {}, add_info
    filtered_pf = cur_data["pf"][can_insert]

    for new_sol, new_obj, new_meas, pf in zip(
        new_data["solution"][can_insert],
        new_data["objective"][can_insert],
        new_data["measures"][can_insert],
        filtered_pf,
    ):
        # Negated because objectives are in [0,100] range and need to be maximized
        # while NonDominatedList minimizes objectives
        # The solutions are actually inserted here instead of at the end of
        # ArrayStore.add() as in vanilla pyribs.
        pf.add(new_sol, -new_obj, new_meas)

    # Pass downstream data for calculating MOQD scores and other archive status updates.
    # Remove duplicate indices (when multiple solutions are added to the same archive cell).
    _, unique_indices = np.unique(indices[can_insert], return_index=True)
    downstream_data = {
        "pf": filtered_pf[unique_indices],
        "hypervolume": list(
            map(lambda pf: pf.hypervolume, filtered_pf[unique_indices])
        ),
    }

    return indices[can_insert][unique_indices], downstream_data, add_info


def compute_moqd_score(indices, new_data, add_info, extra_args, occupied, cur_data):
    """Computes the MOQD score as described in <https://arxiv.org/abs/2202.03057>
    by summing up hypervolumes of Pareto Fronts in every archive cell.

    Empty Pareto Fronts are counted as 0.0.

    Same as compute_objective_sum from vanilla pyribs (see
    <https://github.com/icaros-usc/pyribs/blob/master/ribs/archives/_transforms.py#L226>),
    except the "hypervolume" field instead of "objective" field is summed.
    """
    cur_hypervolume_sum = extra_args["hypervolume_sum"]
    if len(indices) == 0:
        add_info["hypervolume_sum"] = cur_hypervolume_sum
    else:
        cur_hypervolume = cur_data["hypervolume"]
        cur_hypervolume[~occupied] = 0.0
        add_info["hypervolume_sum"] = cur_hypervolume_sum + np.sum(
            new_data["hypervolume"] - cur_hypervolume
        )
    return indices, new_data, add_info


def compute_best_index(indices, new_data, add_info, extra_args, occupied, cur_data):
    """
    Same as compute_best_index from vanilla pyribs (see
    <https://github.com/icaros-usc/pyribs/blob/master/ribs/archives/_transforms.py#L253>),
    except the "hypervolume" field instead of "objective" field is compared.
    """
    # pylint: disable = unused-argument

    if len(indices) == 0:
        add_info["best_index"] = None
    else:
        item_idx = np.argmax(new_data["hypervolume"])
        add_info["best_index"] = indices[item_idx]

    return indices, new_data, add_info


def cvt_archive_heatmap(*args, **kwargs):
    """Same as in vanilla pyribs except uses archive["hypervolume"] for heatmap color
    instead of archive["objective"].
    """
    archive_temp = copy.deepcopy(args[0])
    archive_temp._store._fields["objective"] = archive_temp._store._fields.pop(
        "hypervolume"
    )
    ribs.visualize.cvt_archive_heatmap(archive_temp, *args[1:], **kwargs)
