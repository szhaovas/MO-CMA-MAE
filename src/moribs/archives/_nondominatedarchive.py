#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
from ._hv import HyperVolume
from ._pf_utils import compute_crowding_distances
import itertools
import numpy as np
import logging
from ribs.archives._add_status import AddStatus


logger = logging.getLogger(__name__)


class NonDominatedList(list):
    """
    A list of objective values in an empirical Pareto front,
    meaning that no point strictly domminates another one in all
    objectives.

    >>> from nondominatedarchive import NonDominatedList
    >>> a = NonDominatedList([[1,0.9], [0,1], [0,2]], [2, 2])
    """

    def __init__(
        self,
        init_discount,
        alpha,
        maxlen=None,
        list_of_f_tuples=None,
        reference_point=None,
        seed=None,
    ):
        """
        elements of list_of_f_tuples not in the empirical front are pruned away
        `reference_point` is also used to compute the hypervolume, with the hv
        module of Simon Wessing.

        """
        if list_of_f_tuples is not None and len(list_of_f_tuples):
            try:
                list_of_f_tuples = [tuple(e) for e in list_of_f_tuples]
            except:
                pass
            list.__init__(self, list_of_f_tuples)

        if reference_point is not None:
            self.reference_point = list(reference_point)
        else:
            self.reference_point = reference_point
        self._maxlen = maxlen
        self.prune()  # remove dominated entries, uses self.dominates
        self._hypervolume = None
        self._kink_points = None
        self.solutions = []
        self.measures = []
        self._rng = np.random.default_rng(seed)

        assert 0 < init_discount <= 1
        self._init_discount = init_discount
        self._alpha = alpha
        self._discount_factors = []

        self.numvisits = 0
        self.numdomvisits = 0

    @property
    def objectives(self):
        return [tuple(map(lambda x: -x, row)) for row in self]

    def add(self, solution, f_tuple, measure):
        """add `f_tuple` in `self` if it is not dominated in all objectives."""
        f_objs = -np.array(f_tuple, dtype=np.float64)
        assert np.all(f_objs <= 0)
        cur_objs = np.array(self)
        
        if len(self) == 0:
            discount_factor = self._init_discount
        else:
            cosims = (f_objs @ cur_objs.T) / (np.linalg.norm(f_objs)*np.linalg.norm(cur_objs, axis=1))
            discount_idx = np.argmax(cosims)
            discount_factor = self._discount_factors[discount_idx]
        
        f_objs *= discount_factor

        self.numvisits += 1

        if self.dominates(f_objs):
            return False

        self.numdomvisits += 1
        self.append(f_objs)
        self.solutions.append(solution)
        self.measures.append(measure)
        self._hypervolume = None
        self._kink_points = None

        self._discount_factors.append((1 - self._alpha) * discount_factor + self._alpha)

        self.prune()

        return True

    def remove(self, idx):
        self.pop(idx)
        self.solutions.pop(idx)
        self.measures.pop(idx)
        self._discount_factors.pop(idx)
        self._hypervolume = None
        self._kink_points = None

    def add_list(self, list_of_solutions, list_of_f_tuples, list_of_f_measures):
        """
        add list of f_tuples, not using the add method to avoid calling
        self.prune() several times.

        discount factors are calculated at the start because hypervolume improvements 
        were calculated w.r.t. the discount factors before the add.
        """
        f_objs = -np.array(list_of_f_tuples, dtype=np.float64)
        assert np.all(f_objs <= 0)
        cur_objs = np.array(self)

        batch_size = f_objs.shape[0]
        
        if len(self) == 0:
            discount_factor = np.full((batch_size,1), self._init_discount)
        else:
            cosims = (f_objs @ cur_objs.T) / np.outer(np.linalg.norm(f_objs, axis=1),np.linalg.norm(cur_objs, axis=1))
            discount_idx = np.argmax(cosims, axis=1)
            discount_factor = np.array(self._discount_factors)[discount_idx].reshape((-1,1))
        
        f_objs *= discount_factor

        added = np.full(batch_size, False)
        for i, (sol, obj, meas, dis) in enumerate(zip(
            list_of_solutions, f_objs, list_of_f_measures, discount_factor
        )):
            self.numvisits += 1
            if not self.dominates(obj):
                self.numdomvisits += 1
                self.append(obj)
                self.solutions.append(sol)
                self.measures.append(meas)
                self._hypervolume = None
                self._kink_points = None

                self._discount_factors.append((1 - self._alpha) * dis + self._alpha)

                added[i] = True

        self.prune()

        return added

    def prune(self):
        """
        remove point dominated by another one in all objectives.
        """
        for f_tuple in self:
            if not self.in_domain(f_tuple):
                idx = self.index(f_tuple)
                self.pop(idx)
                self.solutions.pop(idx)
                self.measures.pop(idx)
                self._discount_factors.pop(idx)
        i = 0
        length = len(self)
        while i < length:
            for idx in range(len(self)):
                if np.all(self[idx] == self[i]):
                    continue
                if self.dominates_with(idx, self[i]):
                    del self[i]
                    del self.solutions[i]
                    del self.measures[i]
                    del self._discount_factors[i]
                    i -= 1
                    length -= 1
                    break
            i += 1

        # retains solutions with the largest crowding distances if PF exceeds maxlen
        if not self._maxlen is None:
            # recompute crowding distances after dropping a solution
            while len(self) > self._maxlen:
                crowding_distances = compute_crowding_distances(
                    np.array(self.objectives), boundary_inf=True
                )
                self.remove(np.argsort(crowding_distances)[0])

    def dominates(self, f_tuple):
        """return `True` if any element of `self` dominates or is equal to `f_tuple`.

        Otherwise return `False`.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> a = NDA([[0.39, 0.075], [0.0087, 0.14]])
        >>> a.dominates(a[0])  # is always True if `a` is not empty
        True
        >>> a.dominates([-1, 33]) or a.dominates([33, -1])
        False
        >>> a._asserts()

        """
        if len(self) == 0:
            return False
        for idx in range(len(self)):
            if self.dominates_with(idx, f_tuple):
                return True
        return False

    def dominates_with(self, idx, f_tuple):
        """return `True` if ``self[idx]`` dominates or is equal to `f_tuple`.

        Otherwise return `False` or `None` if `idx` is out-of-range.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> NDA().dominates_with(0, [1, 2]) is None  # empty NDA
        True

        :todo: add more doctests that actually test the functionality and
               not only whether the return value is correct if empty

        """
        if self is None or idx < 0 or idx >= len(self):
            return None
        return self.dominates_with_for(idx, f_tuple)

    # def dominates_with_old(self, idx, f_tuple):
    #     ''' deprecated code, now taken over by dominates_wit_for '''
    #     if all(self[idx][k] <= f_tuple[k] for k in range(len(f_tuple))):
    #         return True
    #     return False

    def dominates_with_for(self, idx, f_tuple):
        """returns true if self[idx] weakly dominates f_tuple

        replaces dominates_with_old because it turned out
        to run quicker
        """
        for k in range(len(f_tuple)):
            if self[idx][k] > f_tuple[k]:
                return False
        else:  # yes, indentation is correct, else is not quite necessary in this case
            return True

    def dominators(self, f_tuple, number_only=False):
        """return the list of all `f_tuple`-dominating elements in `self`,

        including an equal element. ``len(....dominators(...))`` is
        hence the number of dominating elements which can also be obtained
        without creating the list with ``number_only=True``.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> a = NDA([[1.2, 0.1], [0.5, 1]])
        >>> len(a)
        2
        >>> a.dominators([2, 3]) == a
        True
        >>> a.dominators([0.5, 1])
        [(0.5, 1)]
        >>> len(a.dominators([0.6, 3])), a.dominators([0.6, 3], number_only=True)
        (1, 1)
        >>> a.dominators([0.5, 0.9])
        []

        """
        res = 0 if number_only else []
        for idx in range(len(self)):
            if self.dominates_with(idx, f_tuple):
                if number_only:
                    res += 1
                else:
                    res += [self[idx]]
        return res

    def in_domain(self, f_tuple, reference_point=None):
        """return `True` if `f_tuple` is dominating the reference point,

        `False` otherwise. `True` means that `f_tuple` contributes to
        the hypervolume if not dominated by other elements.

        `f_tuple` may also be an index in `self` in which case
        ``self[f_tuple]`` is tested to be in-domain.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> a = NDA([[2.2, 0.1], [0.5, 1]], reference_point=[2, 2])
        >>> assert len(a) == 1
        >>> a.in_domain([0, 0])
        True
        >>> a.in_domain([2, 1])
        False
        >>> all(a.in_domain(ai) for ai in a)
        True
        >>> a.in_domain(0)
        True

        TODO: improve name?
        """
        if reference_point is None:
            reference_point = self.reference_point
        if reference_point is None:
            return True
        try:
            f_tuple = self[f_tuple]
        except TypeError:
            pass
        except IndexError:
            raise  # return None
        if any(
            f_tuple[k] >= reference_point[k]
            for k in range(len(reference_point))
        ):
            return False
        return True

    def _strictly_dominates(self, f_tuple):
        """return `True` if any element of `self` strictly dominates `f_tuple`.

        Otherwise return `False`.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> a = NDA([[0.39, 0.075], [0.0087, 0.14]])
        >>> a.dominates(a[0])  # is always True if `a` is not empty
        True
        >>> a.dominates([-1, 33]) or a.dominates([33, -1])
        False
        >>> a._asserts()

        """
        if len(self) == 0:
            return False
        for idx in range(len(self)):
            if self._strictly_dominates_with(idx, f_tuple):
                return True
        return False

    def _strictly_dominates_with(self, idx, f_tuple):
        """return `True` if ``self[idx]`` strictly dominates `f_tuple`.

        Otherwise return `False` or `None` if `idx` is out-of-range.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> NDA()._strictly_dominates_with(0, [1, 2]) is None  # empty NDA
        True

        """
        if idx < 0 or idx >= len(self):
            return None
        if all(self[idx][k] < f_tuple[k] for k in range(len(f_tuple))):
            return True
        return False

    def _projection(self, f_tuple, i, x):
        length = len(f_tuple)
        res = length * [0]
        res[i] = x
        for j in range(length):
            if j != i:
                res[j] = f_tuple[j]
        return res

    def _projection_to_empirical_front(self, f_tuple):
        """
        return the orthogonal projections of f_tuple on the empirical front,
        with respect to the coodinates axis.
        """
        projections_loose = []
        dominators = self.dominators(f_tuple)
        for point in dominators:
            for i in range(len(f_tuple)):
                projections_loose += [self._projection(f_tuple, i, point[i])]
        projections = []
        for proj in projections_loose:
            if not self._strictly_dominates(proj):
                projections += [proj]
        return projections

    @property
    def kink_points(self):
        """
        Create the 'kink' points from elements of self.
        If f_tuple is not None, also add the projections of f_tuple
        to the empirical front, with respect to the axes
        """

        if self.reference_point is None:
            raise ValueError(
                "to compute the kink points , a reference"
                " point is needed (for the extremal kink points)"
            )
        if self._kink_points is not None:
            return self._kink_points
        kinks_loose = []
        for pair in itertools.combinations(self + [self.reference_point], 2):
            kinks_loose += [[max(x) for x in zip(pair[0], pair[1])]]
        kinks = []
        for kink in kinks_loose:
            if not self._strictly_dominates(kink):
                kinks += [kink]
        self._kink_points = kinks
        return self._kink_points

    @property
    def hypervolume(self):
        """hypervolume of the entire list w.r.t. the "initial" reference point.

        Raise `ValueError` when no reference point was given initially.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> a = NDA([[0.5, 0.4], [0.3, 0.7]], [2, 2.1])
        >>> a._asserts()
        >>> a.reference_point == [2, 2.1]
        True
        >>> a._asserts()

        """
        if self.reference_point is None:
            raise ValueError(
                "to compute the hypervolume a reference"
                " point is needed (must be given initially)"
            )
        if self._hypervolume is None:
            hv_fraction = HyperVolume(self.reference_point)
            self._hypervolume = hv_fraction.compute(self)
        return int(self._hypervolume)

    def contributing_hypervolume(self, f_tuple):
        """
        Hypervolume improvement of f_tuple with respect to self.
        TODO: the argument should be an index, as in moarchiving.
        """
        if self.reference_point is None:
            raise ValueError(
                "to compute the hypervolume a reference"
                " point is needed (must be given initially)"
            )
        hv_fraction = HyperVolume(self.reference_point)
        res1 = hv_fraction.compute(self + [f_tuple])
        res2 = self._hypervolume or hv_fraction.compute(self)
        return res1 - res2

    def distance_to_pareto_front(self, f_tuple):
        """
        Compute the distance of a dominated f_tuple to the empirical Pareto front.
        """
        if self.reference_point is None:
            raise ValueError(
                "to compute the distance to the empirical front"
                "  a reference point is needed (was `None`)"
            )
        if len(self) == 0:
            return (
                sum(
                    [
                        max(0, f_tuple[k] - self.reference_point[k]) ** 2
                        for k in range(len(f_tuple))
                    ]
                )
                ** 0.5
            )
        if not self.dominates(f_tuple):
            if self.in_domain(f_tuple):
                return 0
            return (
                sum(
                    [
                        max(0, f_tuple[k] - self.reference_point[k]) ** 2
                        for k in range(len(f_tuple))
                    ]
                )
                ** 0.5
            )
        squared_distances = []
        for kink in self.kink_points:
            squared_distances += [
                sum((f_tuple[k] - kink[k]) ** 2 for k in range(len(f_tuple)))
            ]
        for proj in self._projection_to_empirical_front(f_tuple):
            squared_distances += [
                sum((f_tuple[k] - proj[k]) ** 2 for k in range(len(f_tuple)))
            ]
        return min(squared_distances) ** 0.5

    def hypervolume_improvement(self, f_tuple):
        """return how much `f_tuple` would improve the hypervolume, and one of three
        status:
            - 0: f_tuple is dominated by the current PF
            - 1: f_tuple is non-dominated by the current PF, and PF is non-empty
            - 2: PF is empty (and thus f_tuple is non-dominated by the current PF)

        If dominated, return the distance to the empirical pareto front
        multiplied by -1.
        Else if not in domain, return distance to the reference point
        dominating area times -1.
        """
        f_objs = -np.array(f_tuple, dtype=np.float64)
        assert np.all(f_objs <= 0)

        contribution = self.contributing_hypervolume(f_objs)
        assert contribution >= 0
        if contribution:
            return (contribution, AddStatus.NEW) if len(self) == 0 else (contribution, AddStatus.IMPROVE_EXISTING)
        return 0, AddStatus.NOT_ADDED

    # @staticmethod
    # def _random_archive(max_size=500, p_ref_point=0.5):
    #     from numpy import random as npr
    #     N = npr.randint(max_size)
    #     ref_point = list(npr.randn(2) + 1) if npr.rand() < p_ref_point else None
    #     return NonDominatedList(
    #         [list(0.01 * npr.randn(2) + npr.rand(1) * [i, -i])
    #          for i in range(N)],
    #         reference_point=ref_point)

    # @staticmethod
    # def _random_archive_many(k, max_size=500, p_ref_point=0.5):
    #     from numpy import random as npr
    #     N = npr.randint(max_size)
    #     ref_point = list(npr.randn(k) + 1) if npr.rand() < p_ref_point else None
    #     return NonDominatedList(
    #         [list(0.01 * npr.randn(k) + i*(2*npr.rand(k)-1))
    #          for i in range(N)],
    #         reference_point=ref_point)

    def _asserts(self):
        """make all kind of consistency assertions.

        >>> import nondominatedarchive
        >>> a = nondominatedarchive.NonDominatedList(
        ...    [[-0.749, -1.188], [-0.557, 1.1076],
        ...    [0.2454, 0.4724], [-1.146, -0.110]], [10, 10])
        >>> a._asserts()
        >>> for p in list(a):
        ...     a.remove(p)
        >>> assert len(a) == 0
        >>> try: a.remove([0, 0])
        ... except ValueError: pass
        ... else: raise AssertionError("remove did not raise ValueError")

        >>> from numpy.random import rand
        >>> for _ in range(120):
        ...     a = nondominatedarchive.NonDominatedList._random_archive()
        ...     if a.reference_point:
        ...         for f_tuple in rand(10, 2):
        ...             h0 = a.hypervolume
        ...             hi = a.hypervolume_improvement(list(f_tuple))
        ...             assert a.hypervolume == h0  # works OK with Fraction

        >>> for _ in range(10):
        ...     for k in range(3,10):
        ...         a = nondominatedarchive.NonDominatedList._random_archive_many(k)
        ...         if a.reference_point:
        ...             for f_tuple in rand(10, k):
        ...                 h0 = a.contributing_hypervolume(list(f_tuple))
        ...                 hi = a.hypervolume_improvement(list(f_tuple))
        ...                 assert h0 >= 0
        ...                 assert h0 == hi or (h0 == 0 and hi < 0)



        """


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    # Example:
    refpoint = [1.1, 1.1, 1.1]
    myfront = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0.25, 0.25, 0.25], [2, 2, 2]]
    emp = NonDominatedList(myfront, 50, refpoint)
