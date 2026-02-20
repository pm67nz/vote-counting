#!/usr/bin/env python3

# Copyright 2023 Peter Maxwell
# This program is free software: you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation, either version 3 of 
# the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.


import numpy as np
from collections import defaultdict

import streamlit as st
import pandas as pd
from io import StringIO

import math

def meek_distribute_votes(ballots, keep_factor):
    """Returns array of vote counts by candidate
    The core of any STV algorithm is to work down each ballot trying to place
    the vote with a candidate.  In the Meek algorithm, each candidate takes  
    some fraction (their Keep Factor) of the votes offered to them this way.
    
    ballots - Collection of sequence of candidate ordinals.
    keep_factor - Sequence of Meek keep factors by candidate ordinal.
     """
    votes = np.zeros_like(keep_factor)
    for (weight, ballot) in ballots:
        remaining_weight = float(weight)
        for candidate in ballot:
            if remaining_weight <= 0: break
            fractional_vote = keep_factor[candidate] * remaining_weight
            votes[candidate] += fractional_vote
            remaining_weight -= fractional_vote
    return votes

def compact_ballots(ballots):
    weights = defaultdict(int)
    for (weight, ballot) in ballots:
        weights[tuple(ballot)] += weight
    ballots = [(weight, list(pattern)) for (pattern, weight) in weights.items()]
    return ballots

def subset_with_weights(ballots, reverse_map, without):
    considered = np.ones([len(reverse_map)], bool)
    considered[without] = False
    growmap = np.where(considered)[0]
    reverse_map = reverse_map[list(growmap)]
    shrinkmap = {b:a for (a,b) in enumerate(growmap)}
    ballots = [(weight, [shrinkmap[b] for b in ballot if considered[b]]) for (weight, ballot) in ballots]
    ballots = compact_ballots(ballots)
    return (ballots, reverse_map)

#@st.cache_data
def meek_se(ballots, withdrawn=[]):
    """List Ranking by Sequential Exclusion using Meek STV with minimal complications,
    so no artificial limits on precision, and no complicated tie-breaking options.
    
    This function handles candidates as integer ordinals, as in the .blt file format,
    So there are no candidate names in here.
    
    ballots - Collection of sequences of candidates.
    withdrawn - Collection of candidates to ignore.
    """

    my_bar = st.progress(0.0, text="Ranking")

    eta = 1e-6
    orig_num_candidates = max(max(b) for (w,b) in ballots) + 1
    order_of_exclusion = []
    reverse_map = np.arange(0, orig_num_candidates, dtype=int)
    excluded = withdrawn
    nvc = orig_num_candidates - len(withdrawn)
    steps_total = nvc**3 / 3 + nvc**2 / 2 + nvc / 6  # Sum of square pyrimidal numbers
    steps_done = 0
    for position in range(nvc, 0, -1):
        # prepare for next round with n-1 candidates
        (ballots, reverse_map) = subset_with_weights(ballots, reverse_map, excluded)
        num_candidates = len(reverse_map)

        hopeful = np.ones([num_candidates], bool)
        elected = np.zeros([num_candidates], bool)
        keep_factor = np.ones([num_candidates], float)
        
        # Find the N-1 winners, one at a time
        seats = position - 1
        for seat in range(seats):
            # Adjust the keep factors of elected candidates
            while True:
                votes = meek_distribute_votes(ballots, keep_factor)
                quota = sum(votes)/(seats+1)
                if all(np.abs(votes[elected]/quota - 1) < eta):
                    break
                keep_factor[elected] = np.minimum(1.0, keep_factor[elected] * quota / votes[elected])
            # Deal with one winner
            candidate = (np.where(hopeful)[0])[np.argmax(votes[hopeful])]
            hopeful[candidate] = False
            elected[candidate] = True
            steps_done += (num_candidates+1)
            my_bar.progress(steps_done / steps_total, text="Ranking")
        
        # The lone remaining candidate is to be excluded
        assert sum(hopeful) == 1
        excluded = np.where(hopeful)[0][0]
        order_of_exclusion.append(reverse_map[excluded])
    
    order_of_exclusion.reverse()
    my_bar.empty()
    return order_of_exclusion

def read_ballots(f):
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
        
    names = [f.readline().strip().replace('"', '') for c in range(candidates)]
    assert '' not in names, 'Nameless candidate(s)'
    assert len(set(names)) == len(names), 'Duplicate candidate names'
    title = f.readline().strip().replace('"', '')
    
    parse_warnings = [f'{flag} ballot lines {msg}' for (flag, msg) in [
            (undervoted, 'included undervotes (skipped ranks). They will be renumbered.'),
            (overvoted, 'included overvotes (tied rankings). They will be tie-broken by order of appearance.'),
            (weighted, 'were weighted, ie: count as multiple ballots.')]
        if flag]

    return (ballots, names, title, parse_warnings, withdrawn, seats)  


st.title('STV-SE ranking')
st.subheader('For generating an ordered Party List from ranked ballots.')
with st.expander("Definition"):
    st.markdown('''STV with Sequential Exclusion can be defined in two ways which turn out to be equivalent:

1) A series of ordinary STV elections for an ever reducing number of seats: 
Given N candidates count the ballots as if for an N-1 seat election, remove the single loser from 
the set of candidates, count the ballots as if for an N-2 seat election, etc.  This is the definition 
given in [Voting Matters 9](https://www.votingmatters.org.uk/ISSUE9/P4.HTM).

2) An STV election in which, whenever a candidate is excluded, the number of seats is decremented 
(and so the quota raised) before continuing.
''')

def calculation_dirty():
    st.session_state.done = False    

def calculation_done():
    st.session_state.done = True    

uploaded_file = st.file_uploader("Choose a ballot file", 
        type=["txt", "blt"], 
        max_upload_size=10, 
        on_change=calculation_dirty,
        help="A [ballot file](https://opavote.com/help/overview#blt-file-format)")

if uploaded_file is not None:
    (ballots, candidates, title, parse_warnings, withdrawn, seats) = read_ballots(
            StringIO(uploaded_file.getvalue().decode("utf-8")))

    if parse_warnings:
        st.warning('\n\n'.join(parse_warnings))

    st.info('"{}"\n\n{} ballots and {} candidates.'.format(title, len(ballots), len(candidates)))

    withdrawn = st.multiselect("Select any candidates who have withdrawn", candidates, on_change=calculation_dirty,
        default = [candidates[w] for w in withdrawn])
    if withdrawn:
        plural = 's' if len(withdrawn) > 1 else ''
        st.info('{} candidate{} withdrawn leaving {} candidates.'.format(len(withdrawn), plural, len(candidates)-len(withdrawn)))
        
    if st.session_state.get("done", False) or st.button("Calculate Ranking", on_click=calculation_done):
        result = meek_se(ballots, withdrawn=[candidates.index(name) for name in withdrawn])

        @st.fragment
        def formatting():
            def store_reserved():
                st.session_state.reserved = st.session_state.reserved_widget
            reserved = st.radio("How many places to add on top for leader(s) etc", [0,1,2], 
                    index = st.session_state.reserved if 'reserved' in st.session_state else 0,
                    key = "reserved_widget", 
                    horizontal = True,
                    on_change = store_reserved)
            
            def text(width, delim):
                lines = [f'{i:{width}}{delim}{candidates[c]}' for (i,c) in enumerate(result, start=reserved+1)]
                return '\n'.join(lines+[''])
            
            with st.container(horizontal=True):
                digits = int(math.log10(len(result)+reserved)) + 1
                st.download_button("Download Text", f'{title}\n\n' + text(digits, '  '), file_name=f'{title} Result.txt', mime='text/plain')
                st.download_button("Download CSV",  'Rank,Candidate\n' + text(1, ','),   file_name=f'{title} Result.csv', mime='text/plain')

            table = pd.DataFrame({'Candidate': [candidates[c] for c in result]}, 
                index = list(range(reserved+1, len(result)+reserved+1)))
            st.table(table)
        formatting()

