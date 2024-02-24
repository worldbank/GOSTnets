import time
import multiprocessing as mp
from pulp import LpInteger, LpVariable, LpProblem, LpMaximize, LpMinimize, lpSum
import pandas

speed_dict = {
    "residential": 20,  # kmph
    "primary": 40,  # kmph
    "primary_link": 35,
    "motorway": 45,
    "motorway_link": 40,
    "trunk": 40,
    "trunk_link": 35,
    "secondary": 30,  # kmph
    "secondary_link": 25,
    "tertiary": 30,
    "tertiary_link": 25,
    "unclassified": 20,
    "road": 20,
    "crossing": 20,
    "living_street": 20,
}


def optimize_facility_locations(
    OD, facilities, p, existing_facilities=None, verbose=False, execute=True, write=""
):
    """
    Function for identifying spatially optimal locations of facilities (P-median problem)

    Parameters
    ----------
    OD : pd.DataFrame
        an Origin:Destination matrix, origins as rows, destinations as columns, in pandas DataFrame format.
    facilities : list
        The 'destinations' of the OD-Matrix. MUST be a list of objects included in OD.columns (or subset) if certain nodes are unsuitable for facility locations
    p : int
        the number of facilities to solve for
    existing_facilities : list
        facilities to always include in the solution. MUST be in 'facilities' list
    verbose : bool
        print a bunch of status updates
    execute : bool
        should the problem be executed
    write : str
        outPath to write problem

    Returns
    -------
    ans : list
        a list of the optimal facility locations

    """
    num_procs = mp.cpu_count()
    if verbose:
        print("cpu count: %s" % num_procs)
    t1 = time.time()

    if type(OD) != pandas.core.frame.DataFrame:
        raise ValueError("OD must be pandas Dataframe!")

    for f in facilities:
        if f not in OD.columns:
            raise ValueError("Potential facility locations MUST be in OD.columns")

    if p < 1:
        raise ValueError("need to solve for more than one facility!")
    elif p > len(facilities):
        raise ValueError("need to solve for fewer locations than location options!")

    if verbose:
        print("Setting up problem")

    origins = OD.index
    origins = list(map(int, origins))

    X = LpVariable.dicts("X", (facilities), 0, 1, LpInteger)

    Y = LpVariable.dicts("Y", (origins, facilities), 0, 1, LpInteger)

    prob = LpProblem("P Median", LpMinimize)

    prob += sum(sum(OD.loc[i, j] * Y[i][j] for j in facilities) for i in origins)

    prob += lpSum([X[j] for j in facilities]) == p

    for i in origins:
        prob += sum(Y[i][j] for j in facilities) == 1

    for i in origins:
        for j in facilities:
            prob += Y[i][j] <= X[j]

    if existing_facilities is not None:
        for e in existing_facilities:
            prob += X[e] == 1

    if verbose:
        print("Set up the problem")

    if write != "":
        prob.writeLP(write)

    if execute:
        prob.solve()
        # the code below needs coin-or to be installed in the OS
        # prob.solve(pulp.COIN_CMD(threads=16)))
        ans = []
        if verbose:
            print("Processing time took: ", time.time() - t1)

        for v in prob.variables():
            subV = v.name.split("_")

            if subV[0] == "X" and v.varValue == 1:
                ans.append(int(str(v).split("_")[1]))

        return ans
    else:
        return prob


def optimize_set_coverage(OD, max_coverage=2000, existing_facilities=None):
    """
    Determine the minimum number of facilities and their locations in order to cover all demands within a pre-specified maximum distance (or time) coverage (Location Set-Covering Problem).

    Parameters
    ----------
    OD : pd.DataFrame
        An Origin:Destination matrix, origins as rows, destinations as columns, in pandas DataFrame format.
    max_coverage : int
        The pre-specified maximum distance (or time) coverage.
    existing_facilities : list
        Facilities to always include in the solution. Must be in 'facilities' list.

    Returns
    -------
    list
        A list of facility locations that provide coverage to all demands within the maximum coverage distance.

    """
    # OD keys must be integers
    OD.columns = OD.columns.astype(int)

    origins = OD.index
    origins = list(map(int, origins))

    facilities = OD.keys()
    facilities = list(map(int, facilities))

    X = LpVariable.dicts("X", (facilities), 0, 1, LpInteger)

    prob = LpProblem("Set Cover", LpMinimize)

    prob += sum(X[j] for j in facilities)

    for i in origins:
        # set of facilities that are eligible to provide coverage to point i
        eligibleFacilities = []
        for j in facilities:
            if OD.loc[i, j] <= max_coverage:
                eligibleFacilities.append(j)
        prob += sum(X[j] for j in eligibleFacilities) >= 1

    prob.solve()

    ans = []

    for v in prob.variables():
        subV = v.name.split("_")

        if subV[0] == "X" and v.varValue == 1:
            ans.append(int(str(v).split("_")[1]))

    if existing_facilities is not None:
        for e in existing_facilities:
            prob += X[e] == 1

    # print out other variables
    print("number of origins")
    print(len(origins))

    totalCoveredFacilities = 0

    for i in origins:
        coveredFacilities = []
        for j in ans:
            if OD.loc[i, j] <= max_coverage:
                coveredFacilities.append(j)
        if len(coveredFacilities) >= 1:
            totalCoveredFacilities += 1

    print("print totalCoveredFacilities")
    print(totalCoveredFacilities)

    print("print percent coverage")
    print(totalCoveredFacilities / len(origins) * 100)

    print("print prob obj")
    print(prob.objective)

    return ans


def optimize_partial_set_coverage(
    OD,
    pop_coverage=0.8,
    max_coverage=2000,
    origins_pop_series=None,
    existing_facilities=None,
):
    """
    Function to determine the minimum number of facilities and their locations in order to cover a given fraction of the population within a pre-specified maximum distance (or time) coverage (Partial Set-Covering Problem). Do not use a demand-weighted OD matrix as an input.

    Parameters
    ----------
    OD : pandas.DataFrame
        An Origin:Destination matrix, origins as rows, destinations as columns.
    pop_coverage : float, optional
        The given fraction of the population that should be covered. Defaults to 0.8.
    max_coverage : int, optional
        The pre-specified maximum distance (or time) coverage. Defaults to 2000.
    origins_pop_series : pandas.Series, optional
        A series that contains each origin as the key, and each origin's population as the value. Defaults to None.
    existing_facilities : list, optional
        Facilities to always include in the solution. Defaults to None.

    Returns
    -------
    list
        A list of facility locations that cover the specified fraction of the population.

    """
    # OD keys must be integers
    OD.columns = OD.columns.astype(int)

    origins = OD.index
    origins = list(map(int, origins))

    facilities = OD.keys()
    facilities = list(map(int, facilities))

    X = LpVariable.dicts("X", (facilities), 0, 1, LpInteger)

    Z = LpVariable.dicts("Z", (origins), 0, 1, LpInteger)

    prob = LpProblem("Partial Set Cover", LpMinimize)

    # objective function
    prob += sum(X[j] for j in facilities)

    for i in origins:
        # set of facilities that are eligible to provide coverage to point i
        eligibleFacilities = []
        for j in facilities:
            if OD.loc[i, j] <= max_coverage:
                eligibleFacilities.append(j)
        # corrected formulation
        prob += sum(X[j] for j in eligibleFacilities) - Z[i] >= 0

    # if origins_pop_series exists then sum up total population and multiply by pop_coverage

    if origins_pop_series is not None:
        # print('print origins_pop_series')
        # print(origins_pop_series)

        # print('print sum(origins_pop_series)')
        # print(sum(origins_pop_series))

        # print('print origins')
        # print(origins)

        # for i in origins:
        # print(i)

        min_coverage = sum(origins_pop_series) * pop_coverage

        print("print min_coverage")
        print(min_coverage)

        prob += sum(Z[i] * origins_pop_series[i] for i in origins) >= min_coverage

    else:
        min_coverage = len(origins) * pop_coverage

        print("print min_coverage")
        print(min_coverage)

        prob += sum(Z[i] for i in origins) >= min_coverage

    if existing_facilities is not None:
        for e in existing_facilities:
            prob += X[e] == 1

    prob.solve()

    ans = []

    for v in prob.variables():
        subV = v.name.split("_")

        if subV[0] == "X" and v.varValue == 1:
            ans.append(int(str(v).split("_")[1]))

    return ans


def optimize_max_coverage(
    OD,
    p_facilities=5,
    max_coverage=2000,
    origins_pop_series=None,
    existing_facilities=None,
):
    """
    Determine the location of P facilities in order to maximize the demand covered within a pre-specified maximum distance coverage (Max Cover).

    Parameters
    ----------
    OD : pandas.DataFrame
        An Origin:Destination matrix, origins as rows, destinations as columns.
    p_facilities : int
        The number of facilities to locate. Default is 5.
    max_coverage : int
        The pre-specified maximum distance (or time) coverage. Default is 2000.
    origins_pop_series : pandas.Series, optional
        Series containing population data for each origin. Default is None.
    existing_facilities : list, optional
        List of facilities to always include in the solution. Default is None.

    Returns
    -------
    list
        List of facility locations that maximize the demand covered.

    """
    # OD keys must be integers
    OD.columns = OD.columns.astype(int)

    origins = OD.index
    origins = list(map(int, origins))

    facilities = OD.keys()
    facilities = list(map(int, facilities))

    # If a facility is located at candidate site j
    X = LpVariable.dicts("X", (facilities), 0, 1, LpInteger)

    # If demand Y is covered
    Y = LpVariable.dicts("Y", (origins), 0, 1, LpInteger)

    prob = LpProblem("Max Cover", LpMaximize)

    if origins_pop_series is not None:
        # objective function
        prob += sum(origins_pop_series[i] * Y[i] for i in origins)

    else:
        # objective function
        prob += sum(Y[i] for i in origins)

    for i in origins:
        # set of facilities that are eligible to provide coverage to point i
        eligibleFacilities = []
        for j in facilities:
            if OD.loc[i, j] <= max_coverage:
                eligibleFacilities.append(j)
        prob += sum(X[j] for j in eligibleFacilities) >= Y[i]

    prob += sum(X[j] for j in facilities) == p_facilities

    if existing_facilities is not None:
        for e in existing_facilities:
            prob += X[e] == 1

    prob.solve()

    # print('print prob')
    # print(prob)

    ans = []

    for v in prob.variables():
        subV = v.name.split("_")

        if subV[0] == "X" and v.varValue == 1:
            ans.append(int(str(v).split("_")[1]))

    print("print objective value")
    print(prob.objective.value())
    # print(prob.objective)

    return ans
