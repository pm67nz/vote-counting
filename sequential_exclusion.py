import numpy as np
from ballots import RankedBallots

def generate_meek_se(ballots: RankedBallots, max_seats=None, withdrawn=[], profile={}, eta=1e-6, compact=True,
        progress_callback = None):
    """List Ranking by Sequential Exclusion using Meek STV with minimal complications,
    so no artificial limits on precision, and no complicated tie-breaking options.
    
    This function handles candidates as integer ordinals, as in the .blt file format,
    So there are no candidate names in here.
    
    ballots - Collection of sequences of candidates.
    withdrawn - Collection of candidates to ignore.
    """

    if progress_callback is None:
        def progress_callback(*args):
            pass

    num_candidates = ballots.num_candidates
    order_of_exclusion = []
    reverse_map = np.arange(0, num_candidates, dtype=int)
    excluded = withdrawn
    nvc = num_candidates - len(withdrawn)
    
    seats = max_seats = min(max_seats or nvc, nvc)
    positions = list(range(nvc, 0, -1))

    efforts = [(p if p <= seats+1 else max(seats**0.5,seats*2-p)/2)**2 for p in range(nvc, 0, -1)] 
    efforts[0] = max(efforts[0], max(efforts)/2)
    steps_total = sum(efforts)
    steps_done = 0

    hopeful = np.ones([num_candidates], bool)
    elected = np.zeros([num_candidates], bool)
    keep_factor = np.ones([num_candidates], float)

    for (position, effort) in zip(positions, efforts):
        seats = min(position - 1, max_seats)
        method = 'STV-SE' if position <= max_seats else 'STV'
        
        if method == 'STV-SE':
            # Reset the winners as we are reducing the number of seats
            hopeful |= elected
            keep_factor[hopeful] = 1.0
            elected[:] = False

        if excluded:
            hopeful[excluded] = False
            keep_factor[excluded] = 0
            num_candidates -= len(excluded)
            if compact:
                # This makes no difference to the result, just an optimisation
                # (with a fairly unimpressive performance benefit).
                (ballots, map_back) = ballots.without(excluded)
                reverse_map = reverse_map[map_back]
                assert num_candidates == len(reverse_map)
                ballots = ballots.compacted()
                keep = hopeful | elected
                hopeful = hopeful[keep]
                elected = elected[keep]
                keep_factor = keep_factor[keep]
            excluded = []
        
        message = f"{method} Electing {seats} out of {num_candidates}"
        steps_done += effort

        while any(hopeful):
            # Adjust the keep factors of elected candidates
            while True:
                votes = ballots.meek_distribute_votes(keep_factor)
                quota = sum(votes)/(seats+1)
                if all(np.abs(votes[elected]/quota - 1) < eta):
                    break
                keep_factor[elected] = np.minimum(1.0, keep_factor[elected] * quota / votes[elected])
            # At least one candidate must now be elected or excluded
            vacancies = seats - elected.sum()
            reached_quota = hopeful.nonzero()[0][votes[hopeful] > quota+eta]
            while reached_quota.size > vacancies:
                reached_quota = np.delete(reached_quota, [])
            if reached_quota.size > 0:
                elected[reached_quota] = True
                hopeful[reached_quota] = False
            else:
                lowest = hopeful.nonzero()[0][np.argmin(votes[hopeful])]
                excluded.append(lowest)
                keep_factor[lowest] = 0
                hopeful[lowest] = False
                progress_callback(steps_done / steps_total, message)
                yield (position, reverse_map[lowest])
                break
