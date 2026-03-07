import networkx as nx
import numpy as np
from ballots import RankedBallots
from time import time
    
def schulze_beatpath(d):    
    # Input: d[i,j], the number of voters who prefer candidate i to candidate j.
    # Output: p[i,j], the strength of the strongest path from candidate i to candidate j.
    C = len(d)
    p = np.zeros_like(d)
    for i in range(C):
        for j in range(C):
            if i != j:
                if d[i,j] > d[j,i]:
                    p[i,j] = d[i,j]

    for i in range(C):
        for j in range(C):
            if i != j:
                for k in range(C):
                    if i != k and j != k:
                        p[j,k] = max(p[j,k], min(p[j,i], p[i,k]))
    order = [(sum(np.sign(p[i,:] - p[:,i])), i) for i in range(C)]
    return [i for (wins, i) in sorted(order, reverse=True)]

def pairwise_schulze(ballots, elected, a, b, eta=1e-3, squash=True):
    # The Schulze Method of Voting
    # Markus Schulze 2018
    # https://arxiv.org/abs/1804.02973
    assert not elected[a] or elected[b]
    assert a != b, (a, b)
    seats = np.count_nonzero(elected) + 1
    
    considered = np.zeros_like(elected)
    considered[elected] = True
    considered[a] = True
    considered[b] = True
    (ballots, map_back, a, b) = ballots.subset(considered, a, b)
     
    # Distribute the votes so as to maximise the minimum over the
    # elected candidates & {a}, with no vote going to a candidate
    # over which b is prefered. By recasting it as a maxflow problem
    # with the constraint that the winners all get the same quota.

    approval_ballots = ballots.approved_over(b)
    
    G = nx.DiGraph()
    G.add_node('source')
    G.add_node('sink')
    #print('  >', a, 'vs', b, 'out of', N)
    for c in range(ballots.num_candidates):
        if c == b: continue
        G.add_node(f'candidate_{c}')
        G.add_edge(f'candidate_{c}', 'sink')
    for (v, (weight, ballot)) in enumerate(approval_ballots):
        G.add_node(f'voter_{v}')
        G.add_edge('source', f'voter_{v}', capacity=weight*1.0)
        for c in ballot:
            #if c == b: break
            G.add_edge(f'voter_{v}', f'candidate_{c}')
    
    r = [approval_ballots.votes / seats]
    while len(r) < 2 or r[-2] - r[-1] > eta:
        for c in range(ballots.num_candidates):
            if c == b: continue
            G.edges[f'candidate_{c}', 'sink']['capacity'] = r[-1]
        flow = nx.maximum_flow_value(G, 'source', 'sink')
        r.append(flow / seats)
    
    #print('    ', len(r), r[:3], r[-1])
    return r[-1]

def schulze_order(ballots: RankedBallots, max_seats=None, withdrawn=[], profile={}, eta=1e-3, compact=True,
        progress_callback=None):
    """List Ranking the method of Markus Schulze.

    This function handles candidates as integer ordinals, as in the .blt file format,
    So there are no candidate names in here.
    
    ballots - Collection of sequences of candidates.
    withdrawn - Collection of candidates to ignore.
    """
    # The Schulze Method of Voting
    # Markus Schulze 2018
    # https://arxiv.org/abs/1804.02973
        
    if progress_callback is None:
        def progress_callback(*args):
            pass
    
    order = []
    nvc = ballots.num_candidates - len(withdrawn)
    max_seats = min(max_seats or nvc, nvc)
    
    pairs = [(remain * remain - remain) for remain in range(nvc-1, nvc-max_seats, -1)]
    effort_per_pair = [(cands * cands) for cands in range(3, max_seats + 2)]
    total_steps = sum(p*e for (p,e) in zip(pairs, effort_per_pair))

    progress = 0
    remaining = [i for i in range(ballots.num_candidates) if i not in withdrawn]
    elected = np.zeros([ballots.num_candidates], bool)

    # top
    condorcet_matrix = ballots.condorcet_matrix()
    winner = schulze_beatpath(condorcet_matrix)[0]
    progress += 1
    progress_callback(progress/total_steps, f"Position 1")
    yield (1, winner)
    elected[winner] = True
    remaining.remove(winner)
    last = progress  
    for (position, epp) in enumerate(effort_per_pair, 2):
        if len(remaining) == 1:
            yield (position, remaining[0])
            break
        condorcet_matrix = np.zeros([len(remaining), len(remaining)], int)
        for (i, a) in enumerate(remaining):
            for (j, b) in enumerate(remaining):
                if i == j: continue
                condorcet_matrix[i,j] = pairwise_schulze(ballots, elected, a, b, eta=eta)
                progress += epp
                if progress > last + 0.01:
                    progress_callback(min(progress/total_steps, 1.0), f'Position {position}')
                    last = progress
        #print(condorcet_matrix)
        winners = schulze_beatpath(condorcet_matrix)
        candidate = remaining[winners[0]]
        order.append(candidate)
        elected[candidate] = True
        progress_callback(min(progress/total_steps, 1.0), f'Position {position}')
        last = progress
        yield (position, candidate)
        remaining.remove(candidate)
    progress_callback(1.0, '')
