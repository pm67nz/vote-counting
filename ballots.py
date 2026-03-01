import numpy as np
from collections import defaultdict
from io import StringIO
import shlex

class Ballots:
    def __init__(self, ballots, num_candidates=None, votes=None):
        self.ballots = tuple((w, self.BALLOT_TYPE(cs)) for (w,cs) in ballots)
        if num_candidates is None:
            num_candidates = max(max(ballot) for (weight, ballot) in self.ballots) + 1
        if votes is None:
            votes = sum(weight for (weight, ballot) in self.ballots)
        self.num_candidates = num_candidates
        self.votes = votes

    def __hash__(self):
        return hash(self.ballots)
        
    def __eq__(self, other):
        return self is other or self.ballots == other.ballots
    
    def __iter__(self):
        return iter(self.ballots)

    def compacted(self):
        packed_ballots = defaultdict(int)
        for (weight, ballot) in self.ballots:
            packed_ballots[ballot] += weight
        ballots = [(weight, ballot) for (ballot, weight) in packed_ballots.items()]
        return type(self)(ballots, self.num_candidates)

    def subset(self, retained, *locate):
        growmap = np.where(retained)[0]
        shrinkmap = {b:a for (a,b) in enumerate(growmap)}
        ballots = [(weight, self.BALLOT_TYPE(shrinkmap[c] for c in ballot if retained[c])) 
                for (weight, ballot) in self.ballots]
        return (type(self)(ballots, retained.sum()), growmap) + tuple(shrinkmap[c] for c in locate)
    
    def without(self, without):
        retained = np.ones([self.num_candidates], bool)
        retained[without] = False
        return self.subset(retained)

class ApprovalBallots(Ballots):
    BALLOT_TYPE = frozenset
    
class RankedBallots(Ballots):
    BALLOT_TYPE = tuple
            
    def meek_distribute_votes(self, keep_factor):
        votes = np.zeros_like(keep_factor)
        for (weight, ballot) in self.ballots:
            remaining_weight = float(weight)
            for candidate in ballot:
                if remaining_weight <= 0: break
                fractional_vote = keep_factor[candidate] * remaining_weight
                votes[candidate] += fractional_vote
                remaining_weight -= fractional_vote
        return votes

    def condorcet_matrix(self):
        d = np.zeros([self.num_candidates, self.num_candidates], int)
        for (weight, ranking) in self.ballots:
            for (rank, candidate) in enumerate(ranking):
                for beaten in ranking[rank+1:]:
                    d[candidate, beaten] += weight
        return d

    def approved_over(self, candidate):
        degenerate_ballots = defaultdict(int)
        for (weight, ranking) in self.ballots:
            trunc = ranking.index(candidate) if candidate in ranking else len(ranking)
            if trunc > 0:
                preferred = frozenset(ranking[:trunc])
                degenerate_ballots[preferred] += weight
        return ApprovalBallots(
            ((weight, preferred) for (preferred, weight) in degenerate_ballots.items()),
            self.num_candidates, self.votes)
        

def parse_ballots(text):
    """Read a .blt format file of ballots.
    File format is 
    <#candidates> <#winners>
    <weight> <1st choice of 1st voter> <2nd choice of 1st voter> ... 0
    <weight> <1st choice of 2nd voter> <2nd choice of 2nd voter> ... 0
    0
    1st candidate name
    2nd candidate name
    Title of the election
    
    All tokens in the first half of the file are integers.
    """
    undervoted = overvoted = weighted = 0
    candidates = seats = None
    withdrawn = []
    ballots = []
    f = StringIO(text)
    while True:
        line = f.readline()
        assert line, 'Unexpected end of file'
        line = line.split('#')[0].strip()
        if '=' in line:
            line = line.replace('=', ' ')
            overvoted += 1
        words = line.split()
        if '-' in words:
            words = [w for w in words if w != '-']
            undervoted += 1
        row = [int(w) for w in words]
        if not row:
            pass
        elif row == [0]:
            break
        elif candidates is None:
            assert len(row) == 2, row
            (candidates, seats) = row
        elif max(row) < 0 and not (ballots or withdrawn):
            assert min(row) < 0, 'Line with only some negative numbers'
            withdrawn.extend(-c-1 for c in row)
        else:
            weight = row.pop(0)
            sentinal = row.pop()
            assert sentinal == 0, f'{sentinal} at end of line'
            assert min(row) > 0, 'Unexpected negative numbers'
            if weight != 1:
                weighted += 1
            ballots.append((weight, [c-1 for c in row]))
        
    names1 = []
    names2 = []
    for line in f.readlines():
        line = line.strip()
        names1.extend([n.strip('"') for n in shlex.split(line, comments=True, posix=False) if n])
        names2.append(line.strip('"'))
    def err(names):
        return abs(len(names) - (candidates + 1))
    names = names1 if err(names1) < err(names2) else names2 
    title = names[candidates]
    names = names[:candidates]
    assert len(set(names)) == len(names), 'Duplicate candidate names'
    
    parse_warnings = [f'{flag} ballot lines {msg}' for (flag, msg) in [
            (undervoted, 'included undervotes (skipped ranks). They will be renumbered.'),
            (overvoted, 'included overvotes (tied rankings). They will be tie-broken by order of appearance.'),
            (weighted, 'were weighted, ie: count as multiple ballots.')]
        if flag]

    return (ballots, names, title, parse_warnings, withdrawn, seats)
